#!/usr/bin/env python3
import os
import json
import logging
import re
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# 22-Notch Rating Map
RATING_MAP = {
    'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18, 'A+': 17, 'A': 16, 'A-': 15,
    'BBB+': 14, 'BBB': 13, 'BBB-': 12, 'BB+': 11, 'BB': 10, 'BB-': 9,
    'B+': 8, 'B': 7, 'B-': 6, 'CCC+': 5, 'CCC': 4, 'CCC-': 3, 'CC': 2, 'C': 1, 'D': 0
}

# 6-Bucket Macro Map
MACRO_MAP = {
    'AAA': 5, 'AA+': 5, 'AA': 5, 'AA-': 5,
    'A+': 4, 'A': 4, 'A-': 4,
    'BBB+': 3, 'BBB': 3, 'BBB-': 3,
    'BB+': 2, 'BB': 2, 'BB-': 2,
    'B+': 1, 'B': 1, 'B-': 1,
    'CCC+': 0, 'CCC': 0, 'CCC-': 0, 'CC': 0, 'C': 0, 'D': 0
}

VALID_RATINGS = list(RATING_MAP.keys())

def get_macro_bucket(rating_str):
    return MACRO_MAP.get(rating_str.strip().upper(), -1)

def get_extracted_text(cik, year, data_dir):
    """
    Load the batch extracted SEC narratives (Items 1, 1A, 7, 7A).
    """
    cik_str = str(cik).zfill(10)
    file_path = data_dir / "egan_sec_extracted_text" / f"{cik_str}_{year}_10-K_extracted.json"
    
    if file_path.exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Combine items 1, 1A, 7, 7A if they exist
            combined = []
            for item in ["1", "1A", "7", "7A"]:
                text = data.get(item, "")
                if text:
                    combined.append(f"--- Item {item} ---\n{text}")
            
            return "\n\n".join(combined)
        except Exception as e:
            logging.warning(f"Failed to read {file_path}: {e}")
    return None

def parse_llm_output(output_text):
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
    for rating in sorted(VALID_RATINGS, key=len, reverse=True):
        if rating in output_text.upper():
            return rating
    return None

def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / 'data'
    out_dir = project_root / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    labels_csv = data_dir / 'egan_training_labels.csv'
    
    if not labels_csv.exists():
        logging.error(f"Missing {labels_csv}. Run extract_egan_labels.py first.")
        return
        
    df = pd.read_csv(labels_csv)
    df['Year'] = df['Year'].astype(str)
    df = df[(df['Rating'] != 'NR')]
    
    api_url = "http://localhost:8000/v1/chat/completions"
    logging.info(f"Using local LLM Server at {api_url}")

    results = []
    y_true_22 = []
    y_pred_22 = []
    y_true_6 = []
    y_pred_6 = []
    
    # Track missing to warn user
    missing_files = 0
    evaluated = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating Egan SEC Filings"):
        cik = row['CIK']
        year = row['Year']
        true_rating_str = row['Rating']
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock

    results_lock = Lock()
    
    def process_sample(row):
        cik = row['CIK']
        year = row['Year']
        true_rating_str = row['Rating']
        
        text = get_extracted_text(cik, year, data_dir)
        if not text:
            return "missing"
            
        system_prompt = "You are an expert corporate credit rating agency. Evaluate the following extracted sections from a company's SEC 10-K and strictly predict the corporate credit rating."
        user_prompt = f"SEC 10-K Extracts:\n{text}\n\nPredict the corporate credit rating. You must select exactly one rating from these options: {', '.join(VALID_RATINGS)}.\nOutput ONLY a valid JSON object in the exact format: {{\"predicted_rating\": \"<rating>\"}}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Drastically reduced from 50k to 15k chars to speed up prefill time
            max_chars = 15000 
            messages[1]["content"] = user_prompt[:max_chars] + ("\n... [truncated]" if len(user_prompt) > max_chars else "")
            
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
            
            response_data = response.json()
            response_text = response_data['choices'][0]['message']['content']
            
        except Exception as e:
            logging.error(f"API Request failed for CIK {cik} Year {year}: {e}")
            return "error"
        
        predicted_rating_str = parse_llm_output(response_text)
        
        if predicted_rating_str and true_rating_str in RATING_MAP:
            with results_lock:
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
            return "success"
        return "error"

    # Run concurrently with up to 16 threads
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_sample, row): idx for idx, row in df.iterrows()}
        for future in tqdm(as_completed(futures), total=len(df), desc="Evaluating Egan SEC Filings"):
            res = future.result()
            if res == "missing":
                missing_files += 1
            elif res == "success":
                evaluated += 1

            
    if missing_files > 0:
        logging.warning(f"Skipped {missing_files} rows because the SEC extracted text file was missing.")
        
    if evaluated == 0:
        logging.error("No samples were successfully evaluated.")
        return
        
    # Calculate and output metrics
    mae_22 = mean_absolute_error(y_true_22, y_pred_22)
    acc_22 = accuracy_score(y_true_22, y_pred_22)
    within_1_22 = np.mean(np.abs(np.array(y_true_22) - np.array(y_pred_22)) <= 1)
    
    mae_6 = mean_absolute_error(y_true_6, y_pred_6)
    acc_6 = accuracy_score(y_true_6, y_pred_6)
    f1_6 = f1_score(y_true_6, y_pred_6, average='weighted')
    
    print("\n" + "="*50)
    print("EGAN DATASET: DIRECT LLM BASELINE METRICS")
    print("="*50)
    print(f"Total Evaluated: {evaluated}")
    print("\n--- 22-Notch Scale ---")
    print(f"Accuracy:         {acc_22:.4f}")
    print(f"Within 1-Notch:   {within_1_22:.4f}")
    print(f"MAE:              {mae_22:.4f}")
    
    print("\n--- 6-Bucket Macro Scale ---")
    print(f"Accuracy:         {acc_6:.4f}")
    print(f"Weighted F1:      {f1_6:.4f}")
    print(f"MAE:              {mae_6:.4f}")
    print("="*50)

    with open(out_dir / 'egan_llm_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
