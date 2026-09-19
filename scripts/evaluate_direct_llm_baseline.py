#!/usr/bin/env python3
import os
import json
import csv
import logging
import re
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# 22-Notch Rating Map
RATING_MAP = {
    'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18, 'A+': 17, 'A': 16, 'A-': 15,
    'BBB+': 14, 'BBB': 13, 'BBB-': 12, 'BB+': 11, 'BB': 10, 'BB-': 9,
    'B+': 8, 'B': 7, 'B-': 6, 'CCC+': 5, 'CCC': 4, 'CCC-': 3, 'CC': 2, 'C': 1, 'D': 0
}

# 6-Bucket Macro Map (Table 2)
MACRO_MAP = {
    'AAA': 5, 'AA+': 5, 'AA': 5, 'AA-': 5,
    'A+': 4, 'A': 4, 'A-': 4,
    'BBB+': 3, 'BBB': 3, 'BBB-': 3,
    'BB+': 2, 'BB': 2, 'BB-': 2,
    'B+': 1, 'B': 1, 'B-': 1,
    'CCC+': 0, 'CCC': 0, 'CCC-': 0, 'CC': 0, 'C': 0, 'D': 0
}

VALID_RATINGS = list(RATING_MAP.keys())

def get_macro_bucket(rating):
    return MACRO_MAP.get(rating, -1)

def get_text_for_sample(cik, year, data_dir):
    """
    Load the extracted qualitative features (RAPTOR summaries) for a given CIK and Year.
    """
    cik_str = str(cik).zfill(10)
    feature_path = data_dir / "qualitative_features" / f"{cik_str}_{year}_features.json"
    
    if feature_path.exists():
        try:
            with open(feature_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Format the JSON nicely as text for the LLM
            formatted_text = []
            if isinstance(data, dict):
                for dim, content in data.items():
                    formatted_text.append(f"--- {dim} ---")
                    if isinstance(content, dict):
                        for k, v in content.items():
                            formatted_text.append(f"{k}: {v}")
                    else:
                        formatted_text.append(str(content))
                    formatted_text.append("")
                return "\n".join(formatted_text)
            else:
                return json.dumps(data, indent=2)
        except Exception as e:
            logging.warning(f"Failed to read {feature_path}: {e}")
            
    return None

def parse_llm_output(output_text):
    """
    Deterministically parse the JSON output to extract the predicted rating.
    Includes fallback heuristics if JSON parsing fails.
    """
    # 1. Try to find a JSON block
    match = re.search(r'\{.*?\}', output_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if "predicted_rating" in parsed:
                pred = str(parsed["predicted_rating"]).strip().upper()
                if pred in RATING_MAP:
                    return pred
        except:
            pass
    
    # 2. Fallback: Search the raw text string for exact rating matches
    # Search from longest to shortest to avoid partial matches (e.g., matching A instead of AAA)
    for rating in sorted(VALID_RATINGS, key=len, reverse=True):
        if rating in output_text.upper():
            return rating
            
    return None

def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / 'data'
    out_dir = project_root / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    kpis_csv = data_dir / 'credit_risk_kpis_master.csv'
    rated_csv = data_dir / 'final_training_labels.csv'
    
    if not kpis_csv.exists() or not rated_csv.exists():
        logging.error(f"Missing required CSV files in {data_dir}. Cannot construct test split.")
        return

    # Reconstruct the CIK Group-Stratified test split (Chronological 80/20)
    logging.info("Reconstructing the exact 20% test split from CredRAG...")
    kpis_df = pd.read_csv(kpis_csv)
    kpis_df['CIK_Identifier'] = kpis_df['CIK_Identifier'].astype(str).str.zfill(10)
    kpis_df['Fiscal_Year'] = kpis_df['Fiscal_Year'].astype(str)
    
    rated_df = pd.read_csv(rated_csv)
    rated_df['CIK'] = rated_df['CIK'].astype(str).str.zfill(10)
    rated_df['Year'] = rated_df['Year'].astype(str)
    
    # Filter valid ratings
    rated_df = rated_df[(rated_df['Rating'] != 'NR')]
    if 'SOURCE_INDICATOR' in rated_df.columns:
        rated_df = rated_df[rated_df['SOURCE_INDICATOR'].astype(str) != '0']
        
    merged_df = pd.merge(rated_df, kpis_df, left_on=['CIK', 'Year'], right_on=['CIK_Identifier', 'Fiscal_Year'], how='inner')
    merged_df = merged_df.sort_values(by=['Year', 'CIK']).reset_index(drop=True)
    
    split_idx = int(len(merged_df) * 0.8)
    test_df = merged_df.iloc[split_idx:].copy()
    logging.info(f"Test split size: {len(test_df)} samples")
    
    # Use local LLM API endpoint
    api_url = "http://localhost:8000/v1/chat/completions"
    logging.info(f"Using local LLM Server at {api_url} (e.g. llama-cpp-python or vLLM)")

    results = []
    y_true_22 = []
    y_pred_22 = []
    y_true_6 = []
    y_pred_6 = []
    
    # Inference loop
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating LLM Baseline"):
        cik = row['CIK']
        year = row['Year']
        true_rating_str = row['Rating']
        
        # Load SEC 10-K Text
        text = get_text_for_sample(cik, year, data_dir)
        if not text:
            # Skip if file not found locally
            continue
            
        system_prompt = "You are an expert corporate credit rating agency. Evaluate the following extracted qualitative summaries from a company's SEC 10-K and strictly predict the corporate credit rating."
        user_prompt = f"Extracted Qualitative Features:\n{text}\n\nPredict the corporate credit rating. You must select exactly one rating from these options: {', '.join(VALID_RATINGS)}.\nOutput ONLY a valid JSON object in the exact format: {{\"predicted_rating\": \"<rating>\"}}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Query the local LLM server
            payload = {
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 20,
            }
            response = requests.post(
                api_url, 
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            # Extract generated response
            response_data = response.json()
            response_text = response_data['choices'][0]['message']['content']
            
        except Exception as e:
            logging.error(f"API Request failed for CIK {cik} Year {year}: {e}")
            continue
        
        # Parse output
        predicted_rating_str = parse_llm_output(response_text)
        
        if predicted_rating_str and true_rating_str in RATING_MAP:
            y_true_22.append(RATING_MAP[true_rating_str])
            y_pred_22.append(RATING_MAP[predicted_rating_str])
            
            y_true_6.append(get_macro_bucket(true_rating_str))
            y_pred_6.append(get_macro_bucket(predicted_rating_str))
            
            results.append({
                "CIK": cik,
                "Year": year,
                "True_Rating": true_rating_str,
                "Predicted_Rating": predicted_rating_str,
                "Raw_LLM_Output": response_text.strip()
            })
            
    # Calculate and output metrics
    if len(y_true_22) > 0:
        y_true_22 = np.array(y_true_22)
        y_pred_22 = np.array(y_pred_22)
        y_true_6 = np.array(y_true_6)
        y_pred_6 = np.array(y_pred_6)
        
        # 22-Notch Metrics
        acc_22 = accuracy_score(y_true_22, y_pred_22)
        mae_22 = mean_absolute_error(y_true_22, y_pred_22)
        f1_22 = f1_score(y_true_22, y_pred_22, average='weighted')
        within_1 = np.mean(np.abs(y_true_22 - y_pred_22) <= 1)
        
        # 6-Bucket Metrics
        acc_6 = accuracy_score(y_true_6, y_pred_6)
        f1_6 = f1_score(y_true_6, y_pred_6, average='weighted')
        
        logging.info("\n========== RESULTS ==========")
        logging.info("--- 22-Notch Scale Metrics ---")
        logging.info(f"Macro Accuracy:      {acc_22:.2%}")
        logging.info(f"MAE (Notches):       {mae_22:.3f}")
        logging.info(f"Within-1-Bucket:     {within_1:.2%}")
        logging.info(f"Weighted F1:         {f1_22:.4f}")
        
        logging.info("\n--- 6-Bucket Macro Scale ---")
        logging.info(f"Macro Accuracy:      {acc_6:.2%}")
        logging.info(f"Weighted F1:         {f1_6:.4f}")
        logging.info("=============================")
        
        metrics_dict = {
            "22_Notch": {
                "Macro_Accuracy": acc_22,
                "MAE": mae_22,
                "Within_1_Bucket": within_1,
                "Weighted_F1": f1_22
            },
            "6_Bucket": {
                "Macro_Accuracy": acc_6,
                "Weighted_F1": f1_6
            }
        }
        
        out_file = out_dir / 'direct_llm_baseline_results.json'
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump({"metrics": metrics_dict, "predictions": results}, f, indent=4)
            
        logging.info(f"Full results and predictions saved to {out_file}")
    else:
        logging.warning("No samples were successfully evaluated. Check if SEC text files exist in the data directory.")

if __name__ == '__main__':
    main()
