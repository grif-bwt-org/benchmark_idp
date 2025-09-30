import os
import json
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any, IO
from PIL import Image, UnidentifiedImageError
import io
from gemini_client import GeminiClient
from logger import default_logger
from validator import validate_invoice_data, ValidatedInvoice

logger = default_logger(__name__)

class DocumentProcessor:
    """
    Processes invoice files (PDFs, images) using the Gemini client
    and saves the structured, validated data to an output directory.
    """
    SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
    SUPPORTED_PDF_EXTENSION = ".pdf"

    MIME_TYPE_MAP = {
        'JPEG': 'image/jpeg',
        'PNG': 'image/png',
        'WEBP': 'image/webp',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp',
    }

    def __init__(
        self,
        gemini_client: GeminiClient,
        output_dir: Union[str, Path],
    ):
        if not isinstance(gemini_client, GeminiClient):
            raise TypeError("gemini_client must be an instance of GeminiClient")

        self.client = gemini_client
        self.output_dir = Path(output_dir).resolve()
        self.processed_files = 0
        self.failed_files = 0
        self._prepare_output_directory()

    def _prepare_output_directory(self) -> bool:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory prepared: {self.output_dir}")
            return True
        except OSError as e:
            logger.critical(f"Failed to create output directory {self.output_dir}: {e}")
            return False

    def _get_image_mime_type(self, image: Image.Image, file_ext: str) -> Optional[str]:
        try:
             mime = image.get_format_mimetype()
             if mime:
                  return mime
        except AttributeError:
             pass

        img_format = image.format
        mime_from_format = self.MIME_TYPE_MAP.get(img_format)
        if mime_from_format:
            return mime_from_format

        logger.debug(f"Could not determine mime type from PIL format '{img_format}'. Falling back to file extension '{file_ext}'.")
        mime_from_ext = self.MIME_TYPE_MAP.get(file_ext.lower())
        if mime_from_ext:
            return mime_from_ext

        logger.warning(f"Could not determine mime type for image format '{img_format}' or extension '{file_ext}'.")
        return None


    def _prepare_file_data(
        self,
        file_stream: IO[bytes],
        filename: str
    ) -> Optional[Tuple[Union[Image.Image, bytes], str]]:

        file_ext = Path(filename).suffix.lower()
        image = None
        try:
            if file_ext in self.SUPPORTED_IMAGE_EXTENSIONS:
                try:
                    image = Image.open(file_stream)
                    image.load()
                    mime_type = self._get_image_mime_type(image, file_ext)
                    if not mime_type:
                        logger.error(f"Could not determine mime type for image: {filename}")
                        if image: image.close()
                        return None
                    logger.debug(f"Successfully prepared image {filename} (format: {image.format}, mode: {image.mode}), determined mime type: {mime_type}")
                    return image, mime_type
                except UnidentifiedImageError:
                    logger.error(f"Cannot identify image file (corrupted/unsupported format): {filename}")
                    if image: image.close()
                    return None
                except IOError as e:
                    logger.error(f"IOError opening or reading image data {filename}: {e}")
                    if image: image.close()
                    return None
                except Exception as e:
                     logger.error(f"Unexpected error processing image data {filename} with PIL: {e}", exc_info=True)
                     if image: image.close()
                     return None

            elif file_ext == self.SUPPORTED_PDF_EXTENSION:
                try:
                    file_stream.seek(0)
                    pdf_bytes = file_stream.read()
                    mime_type = "application/pdf"
                    if not pdf_bytes:
                         logger.error(f"Read 0 bytes from PDF stream: {filename}")
                         return None
                    logger.debug(f"Prepared PDF {filename} ({len(pdf_bytes)} bytes)")
                    return pdf_bytes, mime_type
                except (IOError, Exception) as e:
                    logger.error(f"Error reading PDF data from stream for {filename}: {e}", exc_info=True)
                    return None
            else:
                logger.warning(f"Skipping unsupported file format: {filename} (extension: {file_ext})")
                return None

        except Exception as e:
             logger.error(f"Unexpected error preparing file data for {filename}: {e}", exc_info=True)
             if image: image.close()
             return None

    def _save_result(
        self,
        validated_data: ValidatedInvoice,
        original_filename: str
    ) -> bool:
        if not validated_data:
            logger.error(f"No validated data to save for file {original_filename}.")
            return False
        output_filename = f"{Path(original_filename).stem}.json"
        output_path = self.output_dir / output_filename
        try:
            json_data_to_save = validated_data.model_dump(mode='json')
            with open(output_path, "w", encoding='utf-8') as json_file:
                json.dump(json_data_to_save, json_file, indent=4, ensure_ascii=False)
            logger.info(f"Result for {original_filename} successfully saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error while saving JSON to {output_path}: {e}", exc_info=True)
            return False

    def process_invoice(
        self,
        file_path: Union[str, Path]
    ) -> Optional[Dict[str, Any]]:
        file_path = Path(file_path)
        filename = file_path.name
        logger.info(f"--- Starting processing for file: {filename} ---")
        pil_image = None

        try:
            with open(file_path, 'rb') as f:
                prepare_result = self._prepare_file_data(f, filename)

            if not prepare_result:
                logger.warning(f"Skipping {filename}: failed to prepare file or format is unsupported.")
                self.failed_files += 1
                return None

            invoice_data, mime_type = prepare_result
            if isinstance(invoice_data, Image.Image):
                pil_image = invoice_data

            logger.debug(f"Sending {filename} ({mime_type}) to Gemini...")
            validated_data = self.client.analyze_invoice(invoice_data, mime_type)

            if not validated_data:
                logger.error(f"Gemini (via LangChain) returned no valid data for: {filename}.")
                self.failed_files += 1
                return None
            
            if self._save_result(validated_data, filename):
                self.processed_files += 1
                logger.info(f"Successfully processed, validated, and saved: {filename}.")
                return validated_data.model_dump(mode='json')
            else:
                self.failed_files += 1
                return None

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            self.failed_files += 1
            return None
        except Exception as e:
            logger.error(f"Critical error while processing {filename}: {e}", exc_info=True)
            self.failed_files += 1
            return None
        finally:
            if pil_image:
                pil_image.close()

    def log_summary(self):
        logger.info("=" * 20 + " Processing Summary " + "=" * 20)
        logger.info(f"Successfully processed: {self.processed_files}")
        logger.info(f"Failed files: {self.failed_files}")
        logger.info("=" * 60)

class PdfOnlyInvoiceProcessor:
    def __init__(self, invoice_processor: DocumentProcessor):
        if not isinstance(invoice_processor, DocumentProcessor):
            raise TypeError("invoice_processor must be an instance of DocumentProcessor")
        self.actual_processor = invoice_processor
        self.skipped_non_pdf_files = 0

    def process_invoice(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        file_path = Path(file_path)
        if file_path.suffix.lower() == self.actual_processor.SUPPORTED_PDF_EXTENSION:
            return self.actual_processor.process_invoice(file_path)
        else:
            logger.info(f"[PDF-Filter] Skipping non-PDF file: '{file_path.name}'")
            self.skipped_non_pdf_files += 1
            return None

    def log_summary(self):
        logger.info("=" * 20 + " Overall Summary (PDF-Filter) " + "=" * 20)
        self.actual_processor.log_summary()
        logger.info(f"Skipped non-PDF files: {self.skipped_non_pdf_files}")
        logger.info("=" * 68)
