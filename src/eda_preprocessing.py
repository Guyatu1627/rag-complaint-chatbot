import os
import re
import pandas as pd
import numpy as np

def clean_complaint_text(text):
    """
    Cleans raw customer narratives by lowercasing, removing CFPB 
    anonymization tokens, and stripping boilerplate text.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase conversion
    text = text.lower()
    
    # Remove common boilerplate structures and mask markers (e.g., XX/XX/XXXX or XXXX)
    text = text.replace("i am writing to file a complaint...", "")
    text = re.sub(r'xxxx', '', text)  
    
    # Keep only letters and clean up extra spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def run_preprocessing_pipeline():
    # Define exact directory boundaries
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Look for the raw input file dynamically (checks for .xlsx, .xls, or .csv)
    raw_files = [f for f in os.listdir(raw_dir) if f.endswith(('.xlsx', '.xls', '.csv'))]
    if not raw_files:
        raise FileNotFoundError(f"No raw data file found in '{raw_dir}/'. Please place your complaints file there first!")
    
    input_file_path = os.path.join(raw_dir, raw_files[0])
    print(f"[1/3] Loading raw data from: {input_file_path}")
    
    if input_file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(input_file_path)
    else:
        df = pd.read_csv(input_file_path, low_memory=False)
        
    print(f"Initial raw dataset contains {len(df)} records.")

    # 2. Filter dataset: Retain only the 4 specified products and non-empty narratives
    # (Column name must exactly match the standard CFPB dataset layout)
    target_products = ['Credit Card', 'Personal Loan', 'Savings Account', 'Money Transfer']
    
    print(f"[2/3] Filtering for target products: {target_products}")
    df['Product'] = df['Product'].astype(str).str.strip()
    
    filtered_df = df[
        (df['Product'].isin(target_products)) & 
        (df['Consumer complaint narrative'].notna()) & 
        (df['Consumer complaint narrative'].astype(str).str.strip() != "")
    ].copy()
    
    print(f"Records remaining after target product & non-empty text filter: {len(filtered_df)}")

    # 3. Clean the text narratives
    print("[3/3] Normalizing text narratives and removing boilerplate language...")
    filtered_df['cleaned_narrative'] = filtered_df['Consumer complaint narrative'].apply(clean_complaint_text)
    
    # Drop rows if text became completely blank after cleaning
    filtered_df = filtered_df[filtered_df['cleaned_narrative'].str.strip() != ""]

    # Save out the finished file
    output_csv_path = os.path.join(processed_dir, "filtered_complaints.csv")
    filtered_df.to_csv(output_csv_path, index=False)
    print(f"✔ Success! Cleaned file created at: {output_csv_path} ({len(filtered_df)} records saved)")

if __name__ == "__main__":
    run_preprocessing_pipeline()