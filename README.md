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



# idp_benchmark5

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![AI](https://img.shields.io/badge/AI-LLM%20Powered-purple)
![Status](https://img.shields.io/badge/Status-Active-success)

A comprehensive system for Intelligent Document Processing (IDP) and product recommendation generation. The system automates the process of extracting data from incoming documents (e.g., requests for commercial proposals) and suggests the most relevant products using historical sales data and advanced Language Models (LLMs).

---

## Workflow Overview

The system operates in several key stages:

### 1. Data Extraction
A document (PDF or image) is fed into the `DocumentProcessor`. It uses `GeminiClient` to interact with the Google Gemini API, which extracts structured information (buyer data, line item numbers, part numbers, descriptions).

### 2. Validation
Extracted data undergoes validation using Pydantic models (`validator.py`) to ensure correctness and completeness.

### 3. Candidate Search
For each line item from the document, the `Recommender` performs a fast search through the cached sales history database. It finds an initial list of potential matches by part number and keywords from the description.

### 4. LLM Re-ranking
The best candidates from the previous step are sent to one of the LLM models (Claude, Gemini, or GPT). The model acts as a "procurement expert," thoroughly analyzing each candidate and assigning a final relevance score based on multiple factors (match accuracy, context, supplier frequency, last sale date).

### 5. Export
All results — extracted data, final recommendations with scores and reasoning, as well as intermediate data for analysis — are saved in a structured format to the output directory.

---

## Key Features

-  **Two-stage AI process**: Uses Gemini for fast data extraction and more advanced models (Claude, Gemini, GPT) for expert analysis and ranking
-  **Model flexibility**: Easy switching between different LLMs for recommendations (`recommender_claude.py`, `recommender_gemini.py`, `recommender_gpt.py`)
-  **Advanced NLP processing**: spaCy is used for deep analysis of product descriptions, extracting key tokens and entities
-  **Efficient caching**: Sales history is preprocessed and saved in a fast format (`.parquet`), significantly speeding up subsequent runs
-  **Detailed output for analysis**: The system saves not only final recommendations but all candidates considered by the LLM, along with their scores and reasoning
-  **Performance evaluation**: Automatic calculation of "Hit Rate" metric to assess the accuracy of top recommendations

---

## Project Structure

```
.
├── utils/                   # Helper utilities (logger, validator)
├── gemini_client.py         # Client for Google Gemini API interaction
├── document_processor.py    # Processes input files (PDF, images)
├── main.py                  # Main script orchestrating the entire process
├── recommender_claude.py    # Recommendation logic using Anthropic Claude
├── recommender_gemini.py    # Recommendation logic using Google Gemini
├── recommender_gpt.py       # Recommendation logic using OpenAI GPT
├── exporter.py              # Handles saving all results
├── validator.py             # Contains Pydantic models for data validation
└── requirements.txt         # Project dependencies list
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/grif-bwt-org/benchmark_idp.git
cd benchmark_idp/idp_benchmark5
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv venv

# For Linux/macOS
source venv/bin/activate

# For Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy NLP Model

```bash
python -m spacy download en_core_web_sm
```

### 5. Configure Environment Variables

Create a `.env` file in the project root directory and add your API keys:

```env
GOOGLE_GEMINI_API_KEY="YOUR_GEMINI_KEY"
ANTHROPIC_API_KEY="YOUR_ANTHROPIC_KEY"
OPENAI_API_KEY="YOUR_OPENAI_KEY"
```

---

## Usage

### 1. Prepare Data

- Place your input documents (PDF, JPG, PNG) in the directory specified in `main.py` (default: `with_id/`)
- Place sales history files (in `.csv` format) in the directory specified in `main.py` (default: `data_old/`)

### 2. Select Recommendation Model

In the `main.py` file, change the import line to select the desired Recommender. For example, to use GPT:

```python
# from recommender_claude import Recommender
# from recommender_gemini import Recommender
from recommender_gpt import Recommender
```

### 3. Run the Main Script

```bash
python main.py
```

### 4. Check Results

Results will be saved in the output directory (default: `output_id_llm_claude_with_tokens/`).

For each input file, a separate folder will be created containing:

- `*_extracted.json`: Structured data extracted from the document
- `*_recommendations.json`: Final list of best recommendations
- `*_llm_candidates.csv`: Complete list of candidates analyzed by LLM with their scores and reasoning

---

## Configuration

### Directory Paths
Paths for input data, sales history, and results are configured at the end of the `main.py` file.

### Recommender Parameters
Cutoff thresholds, number of candidates, and other parameters can be configured in the `THRESHOLDS` dictionary inside the respective `recommender_*.py` file.

### Prompts
System and user prompts for LLMs are located inside the `_rerank_with_llm` methods in the `recommender_*.py` files and can be adapted for specific tasks.

---

## Requirements

- Python 3.9 or higher
- Valid API keys for at least one of the supported LLMs
- Sufficient API quota for document processing
- Sales history in CSV format

---

## Supported Formats

### Input Documents
- PDF
- JPG/JPEG
- PNG

### Sales History
- CSV files with previous sales data

---

## Performance Metrics

The system automatically calculates **Hit Rate** — a metric showing how often the correct product appears in the top recommendations.

---

## Support

For questions and issues, please open an issue on the [GitHub repository](https://github.com/grif-bwt-org/benchmark_idp).
