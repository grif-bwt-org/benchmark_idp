import google.generativeai as genai
from typing import Union, Optional, Dict, Any, List
from PIL import Image

from langchain_core.output_parsers import PydanticOutputParser
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from prompts import INVOICE_PROMPT_INSTRUCTIONS
from utils.logger import default_logger
from utils.validator import ValidatedInvoice

logger = default_logger(__name__)

class GeminiClient:
    def __init__(self, api_key: str, default_invoice_prompt: str = INVOICE_PROMPT_INSTRUCTIONS):
        self.default_invoice_prompt = default_invoice_prompt
        self._model_name = 'gemini-2.5-pro'
        
        self._safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        self._default_generation_config = {
            "response_mime_type": "application/json",
        }

        if not api_key:
            raise ValueError("GOOGLE_GEMINI_API_KEY must be provided.")
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                self._model_name,
                safety_settings=self._safety_settings,
                generation_config=self._default_generation_config
            )
            logger.info(f"Hybrid Gemini client initialized with model: {self._model_name}")
            
            self.pydantic_parser = PydanticOutputParser(pydantic_object=ValidatedInvoice)
            self.prompt_with_format_instructions = (
                self.default_invoice_prompt + 
                "\n\n" + 
                self.pydantic_parser.get_format_instructions()
            )

        except Exception as e:
            logger.critical(f"Fatal error configuring Gemini API: {e}")
            raise RuntimeError(f"Failed to configure Gemini API: {e}") from e

    def analyze_invoice(
        self,
        invoice_data: Union[Image.Image, bytes],
        file_mime_type: str
    ) -> Optional[ValidatedInvoice]:
        """
        Analyzes an invoice using the native SDK for the API call and
        LangChain's Pydantic parser for reliable response handling.
        """
        logger.debug(f"Preparing invoice analysis request for mime type: {file_mime_type}")
        
        if isinstance(invoice_data, Image.Image):
             invoice_content_part = invoice_data
        elif isinstance(invoice_data, bytes):
            invoice_content_part = {"mime_type": file_mime_type, "data": invoice_data}
        else:
            logger.error(f"Unsupported invoice data type: {type(invoice_data)}")
            return None
        
        contents = [self.prompt_with_format_instructions, invoice_content_part]

        try:
            logger.info(">>> [API_CALL] Calling generate_content via Gemini")
            response = self.model.generate_content(contents=contents)

            if response.prompt_feedback.block_reason:
                logger.error(f"Gemini API request blocked. Reason: {response.prompt_feedback.block_reason}")
                return None
            
            raw_text_response = response.text
            logger.info(">>> [API_CALL] Received response from Gemini")
            logger.debug(f"Raw text from model:\n{raw_text_response}")
            
            logger.info(">>> [PARSING] Parsing response with LangChain PydanticOutputParser...")
            parsed_object = self.pydantic_parser.parse(raw_text_response)
            
            logger.info(">>> [PARSING] Successfully parsed response into a ValidatedInvoice object.")
            return parsed_object

        except Exception as e:
            logger.error(f"Error during Gemini API call or LangChain parsing: {e}", exc_info=True)
            return None
