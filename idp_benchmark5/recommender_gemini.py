import re
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from thefuzz import fuzz
from datetime import datetime
from collections import defaultdict
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Token
from spacy.util import filter_spans
from utils.logger import default_logger
from dotenv import load_dotenv

try:
    import google.generativeai as genai
except ImportError:
    print("Google Generative AI SDK не установлен. Пожалуйста, установите его: pip install google-generativeai")
    genai = None

logger = default_logger(__name__)


class Recommender:
    COLUMN_MAPPING = {
        'Stk ID': 'part_number', 'Name': 'description', 'Doc Date': 'order_date',
        'Cust Name': 'customer_name', 'Pur Acc ID': 'supplier_name', 'Net Price': 'unit_price',
        'List Price': 'list_price', 'Doc ID': 'doc_id', 'Line Remarks': 'line_remarks'
    }

    THRESHOLDS = {
        "DEFAULT_TOP_N": 5, "DESCRIPTION_SIMILARITY": 60, "CUSTOMER_NAME_CONFIDENCE": 80,
        "MAX_CANDIDATES_TO_SCORE": 2000, "PRICE_ANOMALY_IQR_MULTIPLIER": 2.0, "CATEGORY_TOKENS_COUNT": 2,
        "LLM_CANDIDATE_LIMIT": 30
    }
    INVALID_PART_NUMBERS = {'', 'nan', 'none', 'na', 'n/a', 'tbd', 'tbc', 'без номера'}
    USELESS_POS_TAGS = {"ADV", "ADP", "AUX", "CCONJ", "DET", "INTJ", "PRON", "SCONJ", "SPACE"}
    CACHE_FILENAME = "recommender_cache.parquet"

    def __init__(self, sales_history_dir: Path, cache_dir: Path = Path("cache/")):
        self.history_df: Optional[pd.DataFrame] = None
        self.unique_customers: list = []
        self.nlp = None
        self.matcher = None
        self.llm_model = None

        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            self.matcher = Matcher(self.nlp.vocab)
            patterns = [
                [{"LIKE_NUM": True}, {"LOWER": "x"}, {"LIKE_NUM": True}],
                [{"LIKE_NUM": True}, {"LOWER": "x"}, {"LIKE_NUM": True}, {"LOWER": "x"}, {"LIKE_NUM": True}],
                [{"LIKE_NUM": True}, {"LOWER": {"IN": ["mm", "cm", "m", "in", "ft", "mtr"]}}],
                [{"TEXT": {"REGEX": "^\d+(\.\d+)?$"}}, {"TEXT": "-"}, {"TEXT": {"REGEX": "^\d+(\.\d+)?$"}}]
            ]
            self.matcher.add("COMPOUND_TOKEN", patterns)
            logger.info("spaCy with Matcher loaded successfully.")
        except (ImportError, OSError):
            logger.error("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
        
        load_dotenv()
        api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
        if api_key and genai:
            try:
                genai.configure(api_key=api_key)
                generation_config = genai.types.GenerationConfig(
                    temperature=0.0,
                    top_p=0.95,
                    top_k=40
                )
                self.llm_model = genai.GenerativeModel(
                    "gemini-2.5-pro",
                    generation_config=generation_config
                )
                logger.info("Gemini LLM configured successfully.")
            except Exception as e:
                logger.error(f"Failed to configure Gemini: {e}")
        else:
            logger.warning("GEMINI_API_KEY not found in .env file. LLM-based re-ranking will be disabled.")

        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / self.CACHE_FILENAME
        
        if cache_path.exists():
            try:
                self.history_df = pd.read_parquet(cache_path)
                logger.info(f"Loaded preprocessed data from cache: {cache_path}")
            except Exception as e:
                logger.warning(f"Could not read cache file: {e}. Rebuilding from scratch.")
                self.history_df = self._build_from_scratch(sales_history_dir, cache_path)
        else:
            logger.info("No cache found. Building from scratch.")
            self.history_df = self._build_from_scratch(sales_history_dir, cache_path)

        if self.history_df is not None and not self.history_df.empty:
            for col in ['description_tokens', 'all_pns', 'item_category']:
                 if col in self.history_df.columns:
                    self.history_df[col] = self.history_df[col].apply(
                        lambda x: tuple(x) if isinstance(x, (list, np.ndarray)) else x
                    )
            self.unique_customers = self.history_df['customer_name'].dropna().unique().tolist()

    def _build_from_scratch(self, sales_history_dir: Path, cache_path: Path) -> Optional[pd.DataFrame]:
        df = self._load_history(sales_history_dir)
        if df is None or df.empty: return None
        self._preprocess_data(df)
        try:
            df.to_parquet(cache_path, index=False)
            logger.info(f"Saved preprocessed data to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache file: {e}")
        return df

    def _load_history(self, directory: Path) -> Optional[pd.DataFrame]:
        if not directory.is_dir(): return None
        csv_files = list(directory.glob("*.csv"))
        if not csv_files: return None
        df_list = [self._read_single_csv(f) for f in csv_files if f.stat().st_size > 50]
        df_list = [df for df in df_list if df is not None]
        if not df_list: return None
        full_df = pd.concat(df_list, ignore_index=True)
        return full_df

    def _read_single_csv(self, file_path: Path) -> Optional[pd.DataFrame]:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                header = f.readline().strip().split(';')
            available_cols = [col for col in self.COLUMN_MAPPING.keys() if col in header]
            if not available_cols: return None
            df_single = pd.read_csv(
                file_path, sep=';', usecols=available_cols, low_memory=False, 
                on_bad_lines='skip', encoding='utf-8', dtype=str
            )
            for col in self.COLUMN_MAPPING.keys():
                if col not in df_single.columns: 
                    df_single[self.COLUMN_MAPPING[col]] = None
            df_single.rename(columns=self.COLUMN_MAPPING, inplace=True)
            df_single['source_file'] = file_path.name
            return df_single
        except Exception as e:
            logger.error(f"Failed to read {file_path.name}: {e}", exc_info=True)
            return None

    def _normalize_text_for_spacy(self, text: str) -> str:
        text = text.lower()
        text = text.replace("'", "ft").replace('"', "in")
        text = re.sub(r"[(),/]", " ", text)
        return " ".join(text.split())

    def _is_6_digit_pn(self, token: Token) -> bool:
        return token.is_digit and len(token.text) == 6
        
    def _create_item_category(self, tokens: Tuple[str, ...]) -> Tuple[str, ...]:
        alpha_tokens = sorted([t for t in tokens if t.isalpha()], key=len, reverse=True)
        category = tuple(alpha_tokens[:self.THRESHOLDS["CATEGORY_TOKENS_COUNT"]])
        return category if category else ('__general__',)
    
    def _preprocess_data(self, df: pd.DataFrame):
        logger.info("Starting data preprocessing...")
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        def clean_price_column(price_series: pd.Series) -> pd.Series:
            if price_series is None: return None
            cleaned = price_series.astype(str).str.replace(r'\s+', '', regex=True).str.replace(',', '.', regex=False).str.replace(r'[^\d.]', '', regex=True)
            return pd.to_numeric(cleaned, errors='coerce')
        df['unit_price'] = clean_price_column(df.get('unit_price')).fillna(clean_price_column(df.get('list_price')))
        if 'list_price' in df.columns: df.drop(columns=['list_price'], inplace=True)
        for col in ['part_number', 'description', 'customer_name', 'supplier_name', 'source_file', 'doc_id', 'line_remarks']:
            if col in df.columns: df[col] = df[col].fillna('').str.lower().str.strip()
        df.dropna(subset=['part_number', 'description'], how='all', inplace=True)
        
        if self.nlp:
            logger.info("Processing text with spaCy to extract tokens, PNs and categories...")
            df['full_text_for_nlp'] = df['description'] + " " + df['line_remarks']
            normalized_texts = df['full_text_for_nlp'].apply(self._normalize_text_for_spacy)
            docs = list(self.nlp.pipe(normalized_texts, batch_size=500))
            tokens_list, pns_list, category_list = [], [], []
            for doc in docs:
                tokens, pns = self._get_tokens_and_pns_from_doc(doc)
                tokens_list.append(tokens)
                pns_list.append(pns)
                category_list.append(self._create_item_category(tokens))
            df['description_tokens'] = tokens_list
            df['extracted_pns'] = pns_list
            df['item_category'] = category_list
            df['all_pns'] = df.apply(
                lambda row: tuple(set([pn for pn in [row['part_number']] + list(row['extracted_pns']) if pn and pn not in self.INVALID_PART_NUMBERS])),
                axis=1
            )
            df.drop(columns=['full_text_for_nlp', 'extracted_pns'], inplace=True)
        else: 
            df['description_tokens'] = df['description'].apply(lambda x: tuple(x.split()))
            df['all_pns'] = df['part_number'].apply(lambda x: (x,) if x and x not in self.INVALID_PART_NUMBERS else ())
            df['item_category'] = df['description_tokens'].apply(self._create_item_category)
    
    def _get_tokens_and_pns_from_doc(self, doc: spacy.tokens.Doc) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        matches = self.matcher(doc)
        spans = filter_spans([doc[start:end] for _, start, end in matches])
        with doc.retokenize() as retokenizer:
            for span in spans:
                retokenizer.merge(span)
        desc_tokens, potential_pns = set(), set()
        for token in doc:
            if self._is_6_digit_pn(token):
                potential_pns.add(token.text)
            if token.pos_ not in self.USELESS_POS_TAGS and not token.is_stop and not token.is_punct and len(token.lemma_) > 1:
                desc_tokens.add(token.lemma_.replace(" ", ""))
        return tuple(sorted(list(desc_tokens))), tuple(sorted(list(potential_pns)))
    
    def _get_query_tokens(self, text: str) -> Set[str]:
        if not self.nlp or not text: return set()
        doc = self.nlp(text)
        matches = self.matcher(doc) 
        spans = filter_spans([doc[start:end] for _, start, end in matches])
        with doc.retokenize() as retokenizer:
            for span in spans:
                retokenizer.merge(span)
        return {token.lemma_.replace(" ", "") for token in doc if token.pos_ not in self.USELESS_POS_TAGS and not token.is_stop and not token.is_punct and len(token.lemma_) > 1}
    
    def get_recommendations(self, quotation_data: Dict[str, Any], top_n: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
        if self.history_df is None or self.history_df.empty:
            return {"error": "Sales history not available."}, {}
        if not self.llm_model:
            return {"error": "LLM model not configured. Please check your GEMINI_API_KEY."}, {}

        top_n = top_n if top_n is not None else self.THRESHOLDS["DEFAULT_TOP_N"]
        metadata = quotation_data.get("document_metadata") or {}
        buyer_company = (metadata.get("buyer_company") or "").lower().strip()
        all_recommendations = {}
        
        llm_candidates_for_analysis = {}

        for item in quotation_data.get("line_items", []):
            item_num = item.get("line_item_number")
            if not item_num: continue
            
            candidates_df = self._find_candidates(item)
            if candidates_df.empty:
                all_recommendations[str(item_num)] = []
                continue

            ranked_matches, top_candidates_df = self._rerank_with_llm(item, buyer_company, candidates_df)
 
            if not top_candidates_df.empty:
                llm_candidates_for_analysis[str(item_num)] = top_candidates_df

            final_output = []
            for match in sorted(ranked_matches, key=lambda x: x['score'], reverse=True)[:top_n]:
                date_obj = match['matched_item']['last_order_date']
                match['matched_item']['last_order_date'] = date_obj.strftime('%Y-%m-%d') if pd.notna(date_obj) else None
                match['score'] = round(match['score'], 2)
                final_output.append(match)
            all_recommendations[str(item_num)] = final_output
        
        return all_recommendations, llm_candidates_for_analysis

    def _find_candidates(self, item: Dict[str, Any]) -> pd.DataFrame:
        query_pn = str(item.get("part_number", "")).lower().strip()
        has_valid_pn = query_pn and query_pn not in self.INVALID_PART_NUMBERS
        
        main_desc = (item.get("item_description") or "").strip()
        details_desc = (item.get("item_details") or "").strip()
        all_query_tokens = self._get_query_tokens(self._normalize_text_for_spacy(main_desc + " " + details_desc))

        candidate_df = pd.DataFrame()

        if has_valid_pn:
            mask_pn = self.history_df['all_pns'].apply(lambda pns: query_pn in pns)
            if mask_pn.any():
                candidate_df = self.history_df[mask_pn].copy()
                candidate_df['initial_relevance'] = 200 # Высокий приоритет для PN совпадений
                logger.info(f"Found {len(candidate_df)} primary candidates by Part Number '{query_pn}'.")

        if candidate_df.empty and all_query_tokens:
            if has_valid_pn: logger.info(f"PN '{query_pn}' not found. Falling back to token search.")
            else: logger.info("No valid PN. Searching by tokens.")

            try:

                pattern = '|'.join([re.escape(t) for t in all_query_tokens])
                mask_tokens = self.history_df['description'].str.contains(pattern, case=False, na=False)
                token_candidates_df = self.history_df[mask_tokens].copy()

                if not token_candidates_df.empty:
                    token_candidates_df['initial_relevance'] = token_candidates_df['description_tokens'].apply(
                        lambda hist_tokens: len(all_query_tokens.intersection(set(hist_tokens)))
                    )

                    candidate_df = token_candidates_df[token_candidates_df['initial_relevance'] > 0]
                    logger.info(f"Found {len(candidate_df)} potential token matches.")
            except re.error as e:
                logger.error(f"Regex search for tokens failed: {e}")
                return pd.DataFrame()

        return candidate_df

    def _rerank_with_llm(self, query_item: Dict, buyer_company: str, candidates_df: pd.DataFrame) -> List[Dict]:
        if candidates_df.empty or not self.llm_model:
            return [], pd.DataFrame()

        top_candidates = candidates_df.sort_values(by='initial_relevance', ascending=False).head(self.THRESHOLDS['LLM_CANDIDATE_LIMIT'])

        all_query_tokens = self._get_query_tokens(
            self._normalize_text_for_spacy(
                (query_item.get("item_description") or "") + " " + (query_item.get("item_details") or "")
            )
        )

        request_details = {
            "part_number": query_item.get("part_number"),
            "description": query_item.get("item_description"),
            "details": query_item.get("item_details"),
            "description_keywords": sorted(list(all_query_tokens)),
            "buyer_company": buyer_company
        }
        
        candidate_list = []
        for index, row in top_candidates.iterrows():
            days_since_order = (datetime.now() - row['order_date']).days if pd.notna(row['order_date']) else None
            
            supplier_frequency = (self.history_df['supplier_name'] == row.get('supplier_name')).sum()

            candidate_list.append({
                "candidate_index": index,
                "part_number": row.get('part_number'),
                "description": row.get('description'),
                "line_remarks": row.get('line_remarks'),
                "description_keywords": sorted(list(row.get('description_tokens', []))),
                "unit_price": row.get('unit_price'),
                "last_order_date": row['order_date'].strftime('%Y-%m-%d') if pd.notna(row['order_date']) else None,
                "days_since_last_order": days_since_order,
                "last_customer": row.get('customer_name'),
                "supplier": row.get('supplier_name'),
                "supplier_frequency_in_history": int(supplier_frequency),
                "source_document_type": 'Completed Deal' if 'enqso' in str(row.get('source_file','')).lower() else 'Quotation'
            })

        prompt = f"""
        You are a meticulous purchasing expert analyzing historical sales data to find the best match for a new customer request.
        Your task is to evaluate each candidate item based on a set of precise criteria. For each candidate, you must return a JSON object with your analysis.

        **CRITICAL INSTRUCTION:** Your entire response MUST be a single, valid JSON array of objects. Do not add any text, notes, or markdown formatting before or after the JSON array.

        **EVALUATION CRITERIA & RESPONSE STRUCTURE:**
        For each candidate, provide a JSON object with the following fields:
        - `candidate_index`: The original index of the candidate.
        - `final_score`: Your overall confidence score from 0 (no match) to 100 (perfect match), based on the breakdown.
        - `summary_reason`: A brief, one-sentence summary of your conclusion.
        - `reasoning_breakdown`: An object containing your analysis for each criterion:
            - `part_number_match`: Analyze the part number. Is it an exact match, a match found in the text, or a mismatch?
            - `description_analysis`: Compare the keywords. Note which key terms from the request are present or missing in the candidate. Mention overall similarity.
            - `context_analysis`: Evaluate contextual signals. Comment on the recency (days_since_last_order), the supplier's frequency, and the source document type (a completed deal is a stronger signal).
            - `price_and_customer`: Briefly mention if the price seems reasonable and if the last customer is the same as the current buyer.

        **EXAMPLE RESPONSE OBJECT:**
        {{
          "candidate_index": 591406,
          "final_score": 98,
          "summary_reason": "Exact part number match with identical description and strong contextual signals from a trusted supplier.",
          "reasoning_breakdown": {{
            "part_number_match": "Exact Part Number Match (591406).",
            "description_analysis": "Perfect keyword match: '200mm', 'elec', 'fan', 'phase', 'portable' all present. Description similarity is 100%.",
            "context_analysis": "Source is a 'Quotation', which is a good signal. Last ordered 221 days ago (moderately recent). Supplier 'WAH LEI' is very frequent (appears 500+ times), indicating high trust.",
            "price_and_customer": "Price is available. The last customer is different from the current buyer."
          }}
        }}

        ---
        **USER'S REQUEST:**
        {json.dumps(request_details, indent=2)}

        ---
        **CANDIDATES FROM SALES HISTORY (Evaluate each one):**
        {json.dumps(candidate_list, indent=2)}
        ---

        Now, provide your expert evaluation in the specified JSON array format.
        """

        try:
            logger.debug(f"Sending enhanced prompt to Gemini for item '{query_item.get('line_item_number')}'.")
            response = self.llm_model.generate_content(prompt)
            cleaned_response_text = re.sub(r'```json\s*|\s*```', '', response.text.strip())
            llm_results = json.loads(cleaned_response_text)
            
            scores_map = {res['candidate_index']: res for res in llm_results}

        except (json.JSONDecodeError, AttributeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse enhanced LLM response for item '{query_item.get('line_item_number')}': {e}")
            logger.debug(f"Raw LLM response was: {response.text if 'response' in locals() else 'N/A'}")
            return [], top_candidates
        
        final_results = []
        for index, row in top_candidates.iterrows():
            if index in scores_map:
                llm_info = scores_map[index]
                
                reasons = []
                breakdown = llm_info.get('reasoning_breakdown', {})
                if breakdown.get('part_number_match'): reasons.append(f"PN Match: {breakdown['part_number_match']}")
                if breakdown.get('description_analysis'): reasons.append(f"Description: {breakdown['description_analysis']}")
                if breakdown.get('context_analysis'): reasons.append(f"Context: {breakdown['context_analysis']}")
                if breakdown.get('price_and_customer'): reasons.append(f"Price/Customer: {breakdown['price_and_customer']}")

                if not reasons:
                    reasons.append(llm_info.get('summary_reason', 'No reason provided by LLM.'))

                final_results.append({
                    "score": llm_info.get('final_score', 0),
                    "reasons": reasons,
                    "matched_item": {
                        "part_number": row.get('part_number'),
                        "description": row.get('description'),
                        "unit_price": row.get('unit_price'),
                        "supplier_id": row.get('supplier_name'),
                        "last_order_date": row['order_date'],
                        "customer_name": row.get('customer_name'),
                        "source_doc_id": str(row.get('doc_id', '')).strip() or 'N/A',
                        "match_type": "llm_ranked_detailed" 
                    }
                })
        
        return final_results, top_candidates