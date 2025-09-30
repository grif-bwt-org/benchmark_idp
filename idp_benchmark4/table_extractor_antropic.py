import os
import json
import base64
import logging
import time
import re
from colorlog import ColoredFormatter
from PIL import Image
from prompt import PROMPT_TABLE_EXTRACTION
from io import BytesIO
from anthropic import Anthropic, APITimeoutError
from json_repair import repair_json
import fitz

def setup_logger():
    handler = logging.StreamHandler()
    formatter = ColoredFormatter(
        "%(log_color)s[%(levelname)s] %(message)s",
        log_colors={
            "DEBUG": "cyan", "INFO": "green", "WARNING": "yellow",
            "ERROR": "red", "CRITICAL": "bold_red",
        },
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger

logger = setup_logger()

def log_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

class MockSettings:
    ANTHROPIC_API_KEY = ""

settings = MockSettings()

try:
    if not settings.ANTHROPIC_API_KEY or "xxxxxxxx" in settings.ANTHROPIC_API_KEY:
        logger.critical("API-ключ Anthropic не установлен. Пожалуйста, укажите его в MockSettings.")
        client = None
    else:
        client = Anthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            timeout=1800.0 
        )
        logger.info("Anthropic client initialized successfully with a 30-minute timeout.")
except Exception as e:
    logger.error(f"Failed to initialize Anthropic client: {e}")
    client = None

def convert_pdf_to_images(pdf_path: str, dpi: int = 200) -> list:
    images = []
    try:
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = Image.open(BytesIO(img_data))
            images.append({'image': img, 'page_number': page_num + 1})
        pdf_document.close()
        logger.info(f"Successfully converted PDF '{os.path.basename(pdf_path)}' to {len(images)} image(s)")
        return images
    except Exception as e:
        logger.error(f"Error converting PDF {pdf_path}: {e}")
        return []

def encode_image_to_base64(image: Image.Image) -> str:
    try:
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        return None

@log_execution_time
def analyze_page_with_claude(base64_data: str, filename: str, page_number: int, max_retries: int = 3) -> tuple[dict | None, int, int, int]:
    if not client: return None, 0, 0, 0

    MODEL_NAME = "claude-opus-4-20250514"
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_data,
                    },
                },
                {
                    "type": "text",
                    "text": "Please analyze the attached engineering drawing according to the detailed instructions provided."
                }
            ],
        }
    ]

    for attempt in range(max_retries):
        logger.debug(f"API call attempt {attempt + 1}/{max_retries} for {filename} page {page_number}")
        try:
            response = client.messages.create(
                model=MODEL_NAME,
                system=PROMPT_TABLE_EXTRACTION,
                messages=messages,
                max_tokens=20000,
            )

            if not response.content:
                logger.error(
                    f"API response for {filename} page {page_number} was empty. "
                    f"Stop Reason: '{response.stop_reason}'. This may be due to content filtering."
                )
                if attempt < max_retries - 1:
                    logger.info(f"Retrying after empty response... (attempt {attempt + 2})")
                    time.sleep(5)
                continue

            content = response.content[0].text.strip()
            p_tokens = response.usage.input_tokens
            c_tokens = response.usage.output_tokens
            t_tokens = p_tokens + c_tokens
            logger.info(f"Tokens for {filename} page {page_number}: Prompt={p_tokens}, Completion={c_tokens}, Total={t_tokens}")
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                logger.error(f"No JSON object found in response for {filename} page {page_number}")
                continue
            
            json_string = json_match.group(0)

            try:
                parsed_data = json.loads(json_string)
                logger.info(f"✓ Successfully parsed JSON response for {filename} page {page_number}")
                return parsed_data, p_tokens, c_tokens, t_tokens
            
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error for {filename} page {page_number}: {e}. Attempting to repair.")
                logger.debug(f"Problematic content snippet: {json_string[:500]}...")
                
                try:
                    repaired_json_str = repair_json(json_string)
                    parsed_data = json.loads(repaired_json_str)
                    logger.info(f"✓ Successfully REPAIRED and parsed JSON for {filename} page {page_number}")
                    return parsed_data, p_tokens, c_tokens, t_tokens
                except Exception as repair_e:
                    logger.error(f"Failed to repair JSON for {filename} page {page_number}. Repair error: {repair_e}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying API call... (attempt {attempt + 2})")
                        time.sleep(2)
                    continue

        except APITimeoutError as e:
            logger.error(f"API call timed out for {filename} page {page_number} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying after timeout... (attempt {attempt + 2})")
                time.sleep(5)
            continue
        except Exception as e:
            logger.error(f"API call error for {filename} page {page_number} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying after error... (attempt {attempt + 2})")
                time.sleep(5)
            continue
            
    logger.error(f"All retry attempts failed for {filename} page {page_number}")
    return None, 0, 0, 0

def process_files(input_folder: str, output_folder: str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logger.info(f"Created output directory: {output_folder}")

    supported_extensions = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf")
    files = [f for f in os.listdir(input_folder) if not f.startswith('.') and f.lower().endswith(supported_extensions)]
    logger.info(f"Found {len(files)} files to process")

    for i, filename in enumerate(files, 1):
        logger.info(f"\n{'='*20} Processing file {i}/{len(files)}: {filename} {'='*20}")
        file_path = os.path.join(input_folder, filename)
        base_filename = os.path.splitext(filename)[0]

        images_data = []
        if filename.lower().endswith(".pdf"):
            images_data = convert_pdf_to_images(file_path)
        else:
            try:
                images_data.append({'image': Image.open(file_path), 'page_number': 1})
            except Exception as e:
                logger.error(f"Could not open image file {filename}: {e}")

        if not images_data:
            logger.error(f"✗ No pages/images could be extracted from {filename}. Skipping.")
            continue

        pages_succeeded = 0
        for page in images_data:
            base64_image = encode_image_to_base64(page['image'])
            if not base64_image:
                logger.error(f"Failed to encode page {page['page_number']} of {filename}")
                continue

            parsed_data, _, _, _ = analyze_page_with_claude(
                base64_image, filename, page['page_number']
            )

            if parsed_data:
                pages_succeeded += 1
                output_filename = f"{base_filename}_page_{page['page_number']}.json"
                output_path = os.path.join(output_folder, output_filename)
                try:
                    with open(output_path, "w", encoding='utf-8') as json_file:
                        json.dump(parsed_data, json_file, indent=2, ensure_ascii=False)
                    logger.info(f"✓ Saved results for page {page['page_number']} to: {output_path}")
                except Exception as e:
                    logger.error(f"Failed to save JSON for page {page['page_number']}: {e}")
            else:
                logger.error(f"✗ Failed to analyze page {page['page_number']} of {filename}.")
            
            if len(images_data) > 1: time.sleep(1)

        if pages_succeeded > 0:
            logger.info(f"✓ Finished processing {filename}. Successfully analyzed {pages_succeeded}/{len(images_data)} pages.")
        else:
            logger.error(f"✗ Failed to analyze any pages of {filename}.")

        if i < len(files): time.sleep(2)

    logger.info(f"\n{'='*50}\nPROCESSING COMPLETE\n{'='*50}")

if __name__ == "__main__":
    if not client:
        logger.critical("Anthropic client failed to initialize. Check your API key.")
        exit(1)

    input_folder = "input_folder"
    output_folder = "output_folder"

    if not os.path.exists(input_folder):
        logger.critical(f"Input folder does not exist: {input_folder}")
        exit(1)

    logger.info(f"Starting batch processing...")
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Output folder: {output_folder}")

    try:
        process_files(input_folder, output_folder)
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)