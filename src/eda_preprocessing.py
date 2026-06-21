import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords

# Ensure text processing assets are available locally
nltk.download('stopwords', quiet=True)

def initialize_directories():
    """Creates directory structures if they do not exist."""
    dirs = ['data/raw', 'data/processed', 'notebooks', 'src', 'vector_store', 'tests']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Verified directory: {d}")

def clean_complaint_text(text):
    """
    Cleans raw customer narratives by normalizing text structure,
    stripping punctuation, and dropping common boilerplate phrases.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase conversion
    text = text.lower()
    
    # Remove standard CFPB boilerplate / mask structures
    text = text.replace("i am writing to file a complaint...", "")
    text = re.sub(r'xxxx', '', text)  # Strips out CFPB anonymization tokens (e.g., XX/XX/XXXX)
    
    # Remove special characters, numbers, and extra spacing tabs
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def run_complaint_pipeline(file_path):
    """
    Ingests raw file asset, conducts validation profiling, filters across
    four core financial targets, cleans text fields, and logs metrics.
    """
    print(f"\n[1/4] Loading dataset from {file_path}...")
    # Supports Excel or CSV file types dynamically
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path, low_memory=False)
        
    print(f"Initial Shape: {df.shape[0]} records, {df.shape[1]} columns.")
    
    # Track metrics before filtering
    total_records = len(df)
    has_narrative = df['Consumer complaint narrative'].notna().sum()
    print(f"Data Profile: {has_narrative} entries contain raw narratives ({has_narrative/total_records:.2%}).")
    
    # Define Target Products
    target_products = ['Credit Card', 'Personal Loan', 'Savings Account', 'Money Transfer']
    
    print("\n[2/4] Applying core product filters and removing missing text profiles...")
    # Strip whitespace from column targets to avoid match failures
    df['Product'] = df['Product'].astype(str).str.strip()
    
    # Filter for target products and valid narratives
    filtered_df = df[
        (df['Product'].isin(target_products)) & 
        (df['Consumer complaint narrative'].notna()) & 
        (df['Consumer complaint narrative'].astype(str).str.strip() != "")
    ].copy()
    
    print(f"Filtered Shape: {filtered_df.shape[0]} records remaining.")
    
    print("\n[3/4] Processing narrative lengths and analyzing word counts...")
    filtered_df['word_count'] = filtered_df['Consumer complaint narrative'].apply(lambda x: len(str(x).split()))
    
    # Print profile metrics
    print(f" - Average Narrative Length: {filtered_df['word_count'].mean():.1f} words")
    print(f" - Shortest Narrative Found: {filtered_df['word_count'].min()} words")
    print(f" - Longest Narrative Found: {filtered_df['word_count'].max()} words")
    
    print("\n[4/4] Normalizing narratives and stripping boilerplate text...")
    filtered_df['cleaned_narrative'] = filtered_df['Consumer complaint narrative'].apply(clean_complaint_text)
    
    # Drop records where narrative became completely blank post-cleansing
    filtered_df = filtered_df[filtered_df['cleaned_narrative'].str.strip() != ""]
    
    # Save the output
    output_path = "data/processed/filtered_complaints.csv"
    filtered_df.to_csv(output_path, index=False)
    print(f"✔ Cleaned dataset successfully exported to: {output_path}")
    
    return filtered_df

if __name__ == "__main__":
    initialize_directories()
    # Replace with your actual file name inside data/raw/
    # sample_data_path = "data/raw/complaints.xlsx" 
    # run_complaint_pipeline(sample_data_path)