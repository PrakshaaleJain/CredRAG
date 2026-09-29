import os
import pandas as pd
from pathlib import Path
from datetime import datetime

def main():
    project_root = Path(__file__).resolve().parents[1]
    labels_csv = project_root / 'data' / 'egan_training_labels.csv'
    html_dir = project_root / 'data' / 'egan_sec_filings_html'
    
    if not labels_csv.exists():
        print("Labels CSV not found!")
        return
        
    df = pd.read_csv(labels_csv)
    df['CIK'] = df['CIK'].astype(str).str.zfill(10)
    df['Year'] = df['Year'].astype(str)
    
    matched_files = list(html_dir.glob('*_10-K.html'))
    print(f"Running strict temporal and staleness audit on {len(matched_files)} files...")
    
    valid_count = 0
    stale_count = 0
    leakage_count = 0
    
    for f in matched_files:
        filename = f.stem 
        parts = filename.split('_')
        if len(parts) >= 2:
            cik, year = parts[0], parts[1]
            match = df[(df['CIK'] == cik) & (df['Year'] == year)]
            if not match.empty:
                rating_date_str = match.iloc[0]['Rating_Date']
                rating_date = datetime.strptime(rating_date_str, "%Y-%m-%d")
                
                file_year = int(year)
                rating_year = rating_date.year
                
                if rating_year < file_year:
                    leakage_count += 1
                elif rating_year - file_year > 1:
                    stale_count += 1
                else:
                    valid_count += 1

    print("-" * 40)
    print(f"Strict Audit Results:")
    print(f"  Valid & Timely Pairings : {valid_count}")
    print(f"  Stale Pairings (>1 yr)  : {stale_count}")
    print(f"  Data Leakage (Future)   : {leakage_count}")

if __name__ == '__main__':
    main()