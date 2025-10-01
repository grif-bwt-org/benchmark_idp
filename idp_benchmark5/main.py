import os
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv
import time
import argparse
from datetime import datetime
import pandas as pd 
from typing import Dict

load_dotenv()

from gemini_client import GeminiClient
from document_processor import DocumentProcessor
from recommender_claude import Recommender
from exporter import ResultExporter 
from utils.logger import default_logger

logger = default_logger(__name__)

API_KEY = os.getenv("OPENAI_API_KEY")

def _save_llm_candidates_for_analysis(
    candidates_data: Dict[str, pd.DataFrame],
    output_dir: Path, 
    source_file_stem: str
):
    if not candidates_data:
        logger.warning(f"No LLM candidate data to save for {source_file_stem}.")
        return

    all_dfs = []
    for line_item_num, df in candidates_data.items():
        if not df.empty:
            df_copy = df.copy()
            df_copy['query_line_item'] = line_item_num
            all_dfs.append(df_copy)
    
    if not all_dfs:
        logger.warning(f"No valid LLM candidate DataFrames to concatenate for {source_file_stem}.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    output_path = output_dir / source_file_stem / f"{source_file_stem}_llm_candidates.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved LLM candidates for analysis to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save LLM candidates analysis file: {e}")

def _calculate_hit_rate(recommendations: dict, original_data: dict) -> float:
    """
    Calculates the "Hit Rate" for a single document.
    A "hit" is defined as a line item for which the top recommended candidate
    has the exact same part number as the original request.
    """
    if not recommendations or not original_data.get('line_items'):
        return 0.0
    
    original_pns = {
        str(item.get("line_item_number")): str(item.get("part_number", "")).lower().strip()
        for item in original_data.get("line_items", [])
    }
    
    line_items_with_pn = {k: v for k, v in original_pns.items() if v}
    if not line_items_with_pn:
        return 0.0

    total_countable_items = len(line_items_with_pn)
    hit_count = 0

    for line_num, original_pn in line_items_with_pn.items():
        line_item_candidates = recommendations.get(str(line_num))
        
        if line_item_candidates:
            top_candidate = line_item_candidates[0] 
            recommended_pn = str(top_candidate['matched_item'].get('part_number', "")).lower().strip()
            
            if recommended_pn == original_pn:
                hit_count += 1
    
    return (hit_count / total_countable_items) * 100.0


def main():
    start_time = time.time()
    logger.info("--- Starting batch processing script ---")

    parser = argparse.ArgumentParser(description="Recommendation engine for processing invoices.")
    parser.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--sales_history_dir", type=str, default=str(DEFAULT_SALES_HISTORY_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    sales_history_dir = Path(args.sales_history_dir)
    output_dir = Path(args.output_dir)

    logger.info("Initializing services...")
    try:
        if not API_KEY: raise ValueError("Environment variable GOOGLE_GEMINI_API_KEY not found.")
        if not input_dir.is_dir(): raise FileNotFoundError(f"Input directory not found: {input_dir}")

        gemini_client = GeminiClient(api_key=API_KEY)
        document_processor = DocumentProcessor(gemini_client=gemini_client, output_dir=DEFAULT_OUTPUT_DIR) 
        
        recommender = Recommender(sales_history_dir=sales_history_dir)
        
        exporter = ResultExporter(base_output_dir=output_dir)

        if recommender.history_df is None or recommender.history_df.empty:
            raise RuntimeError(f"Failed to load sales history from {sales_history_dir}. Cannot continue.")
        
        logger.info("Services initialized successfully. Starting file processing...")
    except Exception as e:
        logger.critical(f"Critical error during initialization: {e}", exc_info=True)
        return

    files_to_process = [f for f in input_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
    total_files = len(files_to_process)
    logger.info(f"Found {total_files} files to process in {input_dir}")

    for i, file_path in enumerate(files_to_process, 1):
        logger.info(f"\n[{i}/{total_files}] {'='*20} Processing file: {file_path.name} {'='*20}")
        
        try:
            logger.info("Step 1: Extracting data...")
            parsed_data = document_processor.process_invoice(file_path=file_path)

            if not parsed_data or not parsed_data.get("line_items"):
                logger.error(f"Failed to extract valid data or line items from file: {file_path.name}. Skipping.")
                continue

            logger.info("Step 2: Generating recommendations...")

            recommendations, llm_candidates_data, token_usage_summary = recommender.get_recommendations(parsed_data)

            logger.info(f"LLM Token Usage for '{file_path.name}': Input Tokens = {token_usage_summary['input_tokens']}, Output Tokens = {token_usage_summary['output_tokens']}")
            
            _save_llm_candidates_for_analysis(
                candidates_data=llm_candidates_data,
                output_dir=output_dir,
                source_file_stem=file_path.stem
            )

            try:
                hit_rate = _calculate_hit_rate(recommendations, parsed_data)
            except:
                hit_rate = 0

            logger.info(f"Hit Rate for file '{file_path.name}': {hit_rate:.2f}%")
            
            logger.info("Step 3: Exporting results...")
            exporter.save_all_results(
                source_file_path=file_path,
                parsed_data=parsed_data,
                recommendations=recommendations,
                hit_rate=hit_rate,
            )
            try:
                print(token_usage_summary)
            except:
                print('error for token_usage_summary')

        except Exception as e:
            logger.error(f"An unexpected error occurred while processing file {file_path.name}: {e}", exc_info=True)
            error_result = {"error": f"Critical error: {str(e)}"}
            error_path = output_dir / file_path.stem / f"{file_path.stem}_error.json"
            error_path.parent.mkdir(exist_ok=True)
            with open(error_path, "w", encoding='utf-8') as f:
                json.dump(error_result, f, indent=2, ensure_ascii=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"\n--- Script finished processing {total_files} files in {elapsed_time:.2f} seconds ---")

if __name__ == "__main__":
    logger = default_logger(__name__)
    API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
    
    DEFAULT_INPUT_DIR = Path("with_id/")
    DEFAULT_SALES_HISTORY_DIR = Path("data_old/")
    DEFAULT_OUTPUT_DIR = Path("output_id_llm_claude_with_tokens/")

    main()
