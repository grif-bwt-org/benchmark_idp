import os
import json
import base64
import logging
import time
import re
from colorlog import ColoredFormatter
from prompt import PROMPT_TABLE_EXTRACTION
from PIL import Image
from io import BytesIO
from openai import OpenAI
from json_repair import repair_json

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
    OPENAI_API_KEY = ""

settings = MockSettings()

import fitz

try:
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    logger.info("OpenAI client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    client = None

def convert_pdf_to_images(pdf_path: str, dpi: int = 300) -> list:
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
def analyze_page_with_gpt4v(base64_data: str, filename: str, page_number: int, max_retries: int = 3) -> tuple[dict | None, int, int, int]:
    if not client: return None, 0, 0, 0

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": PROMPT_TABLE_EXTRACTION},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_data}", "detail": "high"}}
        ]}
    ]

    for attempt in range(max_retries):
        logger.debug(f"API call attempt {attempt + 1}/{max_retries} for {filename} page {page_number}")
        try:
            response = client.chat.completions.create(
                model="o3-2025-04-16",
                messages=messages,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            
            p_tokens = response.usage.prompt_tokens if response.usage else 0
            c_tokens = response.usage.completion_tokens if response.usage else 0
            t_tokens = response.usage.total_tokens if response.usage else 0
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
                    repaired_json = repair_json(json_string)
                    parsed_data = json.loads(repaired_json)
                    logger.info(f"✓ Successfully REPAIRED and parsed JSON for {filename} page {page_number}")
                    return parsed_data, p_tokens, c_tokens, t_tokens
                except Exception as repair_e:
                    logger.error(f"Failed to repair JSON for {filename} page {page_number}. Repair error: {repair_e}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying API call... (attempt {attempt + 2})")
                        time.sleep(2)
                        continue
                    return None, p_tokens, c_tokens, t_tokens

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

    session_total_prompt_tokens, session_total_completion_tokens, session_total_tokens = 0, 0, 0
    total_files_succeeded, total_files_failed = 0, 0

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
            total_files_failed += 1
            continue

        file_prompt_tokens, file_completion_tokens, file_total_tokens = 0, 0, 0
        pages_succeeded = 0

        for page in images_data:
            base64_image = encode_image_to_base64(page['image'])
            if not base64_image:
                logger.error(f"Failed to encode page {page['page_number']} of {filename}")
                continue

            parsed_data, p_tokens, c_tokens, t_tokens = analyze_page_with_gpt4v(
                base64_image, filename, page['page_number']
            )

            file_prompt_tokens += p_tokens
            file_completion_tokens += c_tokens
            file_total_tokens += t_tokens

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
            
            if len(images_data) > 1:
                time.sleep(1)

        if pages_succeeded > 0:
            logger.info(f"✓ Finished processing {filename}. Successfully analyzed {pages_succeeded}/{len(images_data)} pages.")
            total_files_succeeded += 1
        else:
            logger.error(f"✗ Failed to analyze any pages of {filename}.")
            total_files_failed += 1

        logger.info(f"Tokens for file {filename}: Prompt={file_prompt_tokens}, Completion={file_completion_tokens}, Total={file_total_tokens}")
        session_total_prompt_tokens += file_prompt_tokens
        session_total_completion_tokens += file_completion_tokens
        session_total_tokens += file_total_tokens

        if i < len(files): time.sleep(2)

    logger.info(f"\n{'='*50}\nPROCESSING COMPLETE\n{'='*50}")
    
if __name__ == "__main__":
    if not client:
        logger.critical("OpenAI client failed to initialize. Check your API key.")
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