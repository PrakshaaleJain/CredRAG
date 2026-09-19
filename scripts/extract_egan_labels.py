#!/usr/bin/env python3
import os
import glob
import xml.etree.ElementTree as ET
import requests
import re
import difflib
import pandas as pd
from datetime import datetime

def normalize_name(name):
    if not name: return ""
    name = name.upper()
    name = re.sub(r'[^A-Z0-9\s]', '', name)
    name = re.sub(r'\b(INC|CORP|CORPORATION|LLC|LP|LTD|PLC|COMPANY|CO)\b', '', name)
    return ' '.join(name.split())

def main():
    headers = {'User-Agent': 'CRED-Rag/1.0 (test@example.com)'}
    print("Fetching SEC ticker-to-CIK mapping...")
    res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
    ticker_to_cik = {}
    name_to_cik = {}
    if res.status_code == 200:
        data = res.json()
        for k, v in data.items():
            cik_str = str(v['cik_str']).zfill(10)
            ticker_to_cik[v['ticker'].upper()] = cik_str
            norm_title = normalize_name(v['title'])
            if norm_title:
                name_to_cik[norm_title] = cik_str
    else:
        print("Failed to fetch SEC tickers.")
        return

    xml_files = glob.glob('data/Egan/*.xml')
    print(f"Found {len(xml_files)} XML files.")
    
    name_keys = list(name_to_cik.keys())
    
    records = []
    
    for i, f in enumerate(xml_files):
        tree = ET.parse(f)
        root = tree.getroot()
        ns = {'r': 'http://xbrl.sec.gov/ratings/2015-03-31'}
        ticker_elem = root.find('.//r:ISI', ns)
        name_elem = root.find('.//r:ISSNAME', ns)
        ticker = ticker_elem.text if ticker_elem is not None else None
        name = name_elem.text if name_elem is not None else None
        
        cik = None
        if ticker and ticker not in ['ENT_01', 'NRSRO'] and ticker.upper() in ticker_to_cik:
            cik = ticker_to_cik[ticker.upper()]
        else:
            norm_name = normalize_name(name)
            if norm_name in name_to_cik:
                cik = name_to_cik[norm_name]
            elif norm_name:
                matches = difflib.get_close_matches(norm_name, name_keys, n=1, cutoff=0.95)
                if matches:
                    cik = name_to_cik[matches[0]]
                    
        if not cik:
            continue
            
        # Parse ratings
        # Find latest long term rating for each year
        year_ratings = {}
        for inrd in root.findall('.//r:ORD', ns) + root.findall('.//r:INRD', ns):
            rad_elem = inrd.find('r:RAD', ns)
            r_elem = inrd.find('r:R', ns)
            rtt_elem = inrd.find('r:RTT', ns) # Short-term or Long-term
            
            if rad_elem is not None and rad_elem.text and r_elem is not None and r_elem.text:
                rtt = rtt_elem.text if rtt_elem is not None else "Long-term"
                if "Short-term" in rtt:
                    continue # Skip short term ratings
                    
                date_str = rad_elem.text
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    year = dt.year
                    rating = r_elem.text
                    
                    if year not in year_ratings:
                        year_ratings[year] = (dt, rating)
                    else:
                        if dt > year_ratings[year][0]:
                            year_ratings[year] = (dt, rating)
                except ValueError:
                    pass
                    
        for year, (dt, rating) in year_ratings.items():
            records.append({
                "CIK": int(cik), # save as int for compatibility with other scripts
                "Year": year,
                "Rating": rating,
                "Company_Name": name,
                "Rating_Date": dt.strftime("%Y-%m-%d")
            })

    df = pd.DataFrame(records)
    out_path = 'data/egan_training_labels.csv'
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} records to {out_path}")

if __name__ == '__main__':
    main()
