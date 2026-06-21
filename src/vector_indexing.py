import os
import pandas as pd
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def run_vector_indexing_pipeline(input_csv_path, sample_size=12000):
    print(f"\n[1/5] Loading cleaned dataset from {input_csv_path}...")
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Cleaned dataset missing at {input_csv_path}. Please execute Task 1 first.")
    
    df = pd.read_csv(input_csv_path)
    print(f"Total processed records available: {len(df)}")
    
    print(f"\n[2/5] Executing Proportional Stratified Sampling (Target: {sample_size} records)...")
    # Determine proportion weights per product channel
    product_counts = df['Product'].value_counts(normalize=True)
    print("Base distribution ratios per product:")
    for prod, ratio in product_counts.items():
        print(f" - {prod}: {ratio:.2%}")
        
    # Apply stratified extraction logic safely
    sampled_df = df.groupby('Product', group_keys=False).apply(
        lambda x: x.sample(int(np.ceil(len(x) * (sample_size / len(df)))), random_state=42)
    ).sample(n=sample_size, random_state=42) # Re-shuffle and lock down exact count
    
    print(f"Stratified sample locked at {len(sampled_df)} records.")
    
    print("\n[3/5] Initializing Text Chunking Strategy (Recursive Character Splitting)...")
    # Configuration chosen to balance operational granularity with safe context windows
    chunk_size = 500
    chunk_overlap = 50
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    print("\n[4/5] Constructing Document Objects with Traceable Source Metadata...")
    documents = []
    # Use the raw narrative for semantic embedding clarity or cleaned narrative depending on model preference
    # We use 'cleaned_narrative' for pure semantic signal matching
    for _, row in sampled_df.iterrows():
        narrative_text = str(row['cleaned_narrative'])
        
        # Split text into manageable structural pieces
        chunks = text_splitter.split_text(narrative_text)
        
        # Capture trace indicators back to the original database row
        for i, chunk in enumerate(chunks):
            metadata = {
                "complaint_id": str(row.get('Complaint ID', 'UNKNOWN')),
                "product": str(row['Product']),
                "chunk_index": i
            }
            documents.append(Document(page_content=chunk, metadata=metadata))
            
    print(f"Generated {len(documents)} text chunks from {len(sampled_df)} records.")
    
    print("\n[5/5] Downloading Embedding Model & Generating FAISS Vector Index Space...")
    # Sentence-Transformers/all-MiniLM-L6-v2 runs locally and efficiently maps 384-dimension tensors
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    encode_kwargs = {'normalize_embeddings': True} # Ensures cosine similarity equivalence during search
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs
    )
    
    print("Processing vector allocations (this may take a few minutes depending on CPU performance)...")
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Save the local index database parameters to disk
    output_dir = "vector_store"
    os.makedirs(output_dir, exist_ok=True)
    vector_store.save_local(os.path.join(output_dir, "faiss_index"))
    print(f"✔ Success! Persisted vector database index securely to: {output_dir}/faiss_index")

if __name__ == "__main__":
    # Point path to the file exported during Task 1 processing
    cleaned_data_input = "data/processed/filtered_complaints.csv"
    run_vector_indexing_pipeline(cleaned_data_input, sample_size=12000)