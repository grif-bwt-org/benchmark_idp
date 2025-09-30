import os
import json
import logging
import time 
from typing import Union, Optional
from prompt import PROMPT_TABLE_EXTRACTION
from pathlib import Path
from colorlog import ColoredFormatter
from PIL import Image, UnidentifiedImageError 
from utils.logger import log_execution_time
from settings.config import settings
import google.generativeai as genai
from google.api_core import exceptions
from google.generativeai.types import HarmCategory, HarmBlockThreshold

handler = logging.StreamHandler()
formatter = ColoredFormatter(
    "%(log_color)s[%(levelname)s] %(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

try:
    genai.configure(api_key=settings.GOOGLE_GEMINI_API_KEY)
except AttributeError:
    logger.critical("GOOGLE_GEMINI_API_KEY not found in settings.config!")
    pass
except Exception as e:
    logger.critical(f"Error configuring Gemini API: {e}")

MODEL_NAME = 'gemini-2.5-flash'


SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

GENERATION_CONFIG = {
    "response_mime_type": "application/json", 
}

@log_execution_time
def analyze_invoice_to_dict_gemini(
    invoice_data: Union[Image.Image, bytes],
    file_mime_type: str,
    max_retries: int = 3,
    retry_delay_seconds: int = 5
) -> Optional[dict]:
    """
    Analyze an invoice image or PDF using the Gemini API with retries and timeouts.
    ... (docstring) ...
    """
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(
                MODEL_NAME
            )

            logger.info(f"Starting analysis for invoice data (type: {file_mime_type}). Attempt {attempt + 1} of {max_retries}.")

            if isinstance(invoice_data, Image.Image):
                invoice_content_part = invoice_data
            elif isinstance(invoice_data, bytes):
                invoice_content_part = {"mime_type": file_mime_type, "data": invoice_data}
            else:
                logger.error(f"Unsupported data type provided: {type(invoice_data)}")
                return None

            contents = [PROMPT_TABLE_EXTRACTION, invoice_content_part]

            request_options = {"timeout": 300}

            response = model.generate_content(
                contents=contents,
                generation_config=GENERATION_CONFIG,
                safety_settings=SAFETY_SETTINGS,
                request_options=request_options
            )

            logger.debug(f"Gemini API raw response parts: {response.parts}")
            result_text = response.text.strip()

            if not result_text:
                if response.prompt_feedback.block_reason:
                    logger.error(f"Gemini API call blocked. Reason: {response.prompt_feedback.block_reason}")
                    logger.error(f"Block details: {response.prompt_feedback.block_reason_message}")
                else:
                    logger.error("Gemini API returned an empty response.")
                return None
            
            if result_text.startswith("```json"):
                result_text = result_text[7:-3].strip()

            parsed_result = json.loads(result_text)
            logger.info("Successfully parsed result.")
            return parsed_result

        except exceptions.InternalServerError as e:
            logger.warning(f"Attempt {attempt + 1} failed with Internal Server Error (500): {e}")
            if attempt + 1 == max_retries:
                logger.error("Max retries reached. Failing the operation for this file.")
                return None
            logger.info(f"Waiting {retry_delay_seconds} seconds before retrying...")
            time.sleep(retry_delay_seconds)

        except exceptions.DeadlineExceeded as e:
            logger.warning(f"Attempt {attempt + 1} failed with Deadline Exceeded (timeout): {e}")
            if attempt + 1 == max_retries:
                logger.error("Max retries reached after timeouts. Failing the operation for this file.")
                return None
            logger.info(f"Waiting {retry_delay_seconds} seconds before retrying...")
            time.sleep(retry_delay_seconds)

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}. Raw response text: '{result_text}'")
            return None
            
        except AttributeError as e:
            if "GOOGLE_GEMINI_API_KEY" in str(e) or "'NoneType' object has no attribute 'GenerativeModel'" in str(e):
                 logger.critical(f"Gemini API key might be missing or invalid: {e}")
            else:
                 logger.error(f"An unexpected attribute error occurred: {e}")
            logger.exception("Detailed exception information:")
            return None
            
        except Exception as e:
            logger.error(f"An unhandled error occurred during Gemini API request or processing: {e}")
            logger.exception("Detailed exception information:")
            return None 

    return None


if __name__ == "__main__":
    input_folder = Path("input_folder")
    output_folder = Path("output_folder")

    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}

    pdf_extension = ".pdf"

    if not input_folder.is_dir():
        logger.critical(f"Input folder does not exist: {input_folder}")
        exit(1)

    output_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting invoice processing from folder: {input_folder}")
    logger.info(f"Output will be saved to: {output_folder}")
    logger.info(f"Using Gemini model: {MODEL_NAME}")


    processed_files = 0
    failed_files = 0

    for item in input_folder.iterdir():
        if not item.is_file():
            logger.debug(f"Skipping non-file item: {item.name}")
            continue

        file_path = item
        file_ext = file_path.suffix.lower()
        filename = file_path.name

        invoice_data_to_process = None
        mime_type = None

        try:
            if file_ext in image_extensions:
                logger.info(f"Processing image file: {filename}")
                try:
                    image = Image.open(file_path)
                    image.load()
                    invoice_data_to_process = image
                    img_format = image.format
                    if img_format == 'JPEG':
                        mime_type = 'image/jpeg'
                    elif img_format == 'PNG':
                        mime_type = 'image/png'
                    elif img_format == 'WEBP':
                         mime_type = 'image/webp'
                    else:
                         logger.warning(f"Could not determine specific image mime type for {filename}, using generic.")
                         if file_ext == ".jpg" or file_ext == ".jpeg": mime_type = "image/jpeg"
                         elif file_ext == ".png": mime_type = "image/png"
                         elif file_ext == ".webp": mime_type = "image/webp"
                         else:
                             logger.error(f"Unsupported image format based on extension: {filename}")
                             failed_files += 1
                             continue

                except UnidentifiedImageError:
                    logger.error(f"Cannot identify image file (corrupted or unsupported format): {filename}")
                    failed_filess += 1
                    continue
                except IOError as e:
                    logger.error(f"Could not open or read image file {filename}: {e}")
                    failed_files += 1
                    continue

            elif file_ext == pdf_extension:
                logger.info(f"Processing PDF file: {filename}")
                try:
                    with open(file_path, "rb") as f:
                        pdf_bytes = f.read()
                    invoice_data_to_process = pdf_bytes
                    mime_type = "application/pdf"
                except IOError as e:
                    logger.error(f"Could not read PDF file {filename}: {e}")
                    failed_files += 1
                    continue

            else:
                logger.warning(f"Skipping unsupported file format: {filename} (extension: {file_ext})")
                continue

            if invoice_data_to_process and mime_type:
                parsed_invoice = analyze_invoice_to_dict_gemini(invoice_data_to_process, mime_type)

                if parsed_invoice:
                    output_path = output_folder / f"{file_path.stem}.json"
                    try:
                        with open(output_path, "w", encoding='utf-8') as json_file:
                            json.dump(parsed_invoice, json_file, indent=4, ensure_ascii=False)
                        logger.info(f"Saved parsed data to {output_path}")
                        processed_files += 1
                    except IOError as e:
                        logger.error(f"Could not write JSON output file {output_path}: {e}")
                        failed_files += 1
                    except TypeError as e:
                         logger.error(f"Data serialization error for {filename}: {e}. Parsed data might not be JSON serializable.")
                         failed_files += 1

                else:
                    logger.error(f"Failed to parse the invoice: {filename}")
                    failed_files += 1

                if isinstance(invoice_data_to_process, Image.Image):
                    invoice_data_to_process.close()


        except Exception as e:
            logger.error(f"An unexpected error occurred processing {filename}: {e}")
            logger.exception("Detailed exception information for file processing:")
            failed_files += 1
            if 'image' in locals() and isinstance(image, Image.Image):
                 try:
                     image.close()
                 except Exception:
                     pass


    logger.info("="*20 + " Processing Summary " + "="*20)
    logger.info(f"Total files processed successfully: {processed_files}")
    logger.info(f"Total files failed: {failed_files}")
    logger.info("="*58)