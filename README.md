# idp_benchmark4

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)  
![Status](https://img.shields.io/badge/Status-Active-success)

This project provides an AI-powered tool for extracting structured data from **engineering drawings** (blueprints) in PDF and image formats. It supports Anthropic Claude, OpenAI ChatGPT, and Google Gemini.

---

## Features

- Optimized for multi-sheet engineering drawings
- Multi-model support (Claude, ChatGPT, Gemini)
- Works with PDFs and images (PNG, JPG, JPEG, WEBP, GIF)
- Batch processing for multiple files
- JSON output aligned with engineering view/section details
- Configurable prompts and detailed logging

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/grif-bwt-org/benchmark_idp.git
cd benchmark_idp/idp_benchmark4
pip install -r requirements.txt
```

### 2. Configure API Keys

Set API keys in the respective script:

- `ANTHROPIC_API_KEY` → `table_extractor_antropic.py`
- `OPENAI_API_KEY` → `table_extractor_chatgpt.py`
- `GOOGLE_GEMINI_API_KEY` → `settings.config`

### 3. Prepare Input Files

Place engineering drawings (PDF or images) in `input_folder/`.

---

## Run Extraction

Execute the script corresponding to the AI model you want to use:

```bash
# For ChatGPT
python table_extractor_chatgpt.py

# For Anthropic Claude
python table_extractor_antropic.py

# For Google Gemini
python table_extractor_gemini.py
```

After execution, the extracted data will be saved as JSON files in the `output_folder/`.

---

## Configuration

### Extraction Prompt

Tuned for engineering drawings. Example: see `PROMPT_TABLE_EXTRACTION` in `prompt.py`.

### Folder Paths

Input and output folders can be adjusted in the `__main__` section of each script.

### Model Selection

Change the `MODEL_NAME` variable inside the script to switch between different AI models.

---

## Project Structure

```
idp_benchmark4/
├── input_folder/          # Place your engineering drawings here
├── output_folder/         # Extracted JSON files will be saved here
├── table_extractor_chatgpt.py
├── table_extractor_antropic.py
├── table_extractor_gemini.py
├── prompt.py              # Extraction prompts configuration
├── settings.config        # API keys and settings
└── requirements.txt       # Python dependencies
```

---

## Output Format

The tool generates JSON files with structured data extracted from engineering drawings, including:

- View/section identifiers
- Technical specifications
- Measurements and dimensions
- Material information
- Other relevant engineering data

---

## Requirements

- Python 3.9 or higher
- Valid API keys for at least one of the supported AI models
- Sufficient API credits/quota for processing

---

## Support

For issues and questions, please open an issue on the [GitHub repository](https://github.com/grif-bwt-org/benchmark_idp).
