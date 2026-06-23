import os
import re
import pandas as pd

def clean_complaint_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(
        r"i am writing to file a complaint regarding|dear customer support",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"x{2,}", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9\s\.,!\?]", "", text)
    text = " ".join(text.split()).lower()
    return text

def execute_preprocessing_pipeline(input_path, output_path):
    print("Executing Task 1: Preprocessing Data Stream...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing input source dataset at {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    df = df.dropna(subset=["Consumer complaint narrative"])

    target_products = ["Credit card", "Personal loan", "Savings account", "Money transfer"]
    product_map = {p.lower(): p for p in target_products}
    df["Product"] = df["Product"].astype(str).str.strip()
    df["_product_key"] = df["Product"].str.lower()
    df = df[df["_product_key"].isin(product_map)].copy()
    df["Product"] = df["_product_key"].map(product_map)
    df = df.drop(columns=["_product_key"])

    df["cleaned_narrative"] = df["Consumer complaint narrative"].apply(clean_complaint_text)
    df = df[df["cleaned_narrative"] != ""]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Task 1 Complete. Preprocessed records exported to: {output_path}")

if __name__ == "__main__":
    raw_data_path = "data/raw/complaints.csv"
    processed_data_path = "data/processed/filtered_complaints.csv"

    if not os.path.exists(raw_data_path):
        os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
        mock_df = pd.DataFrame(
            {
                "Complaint ID": [101, 102, 103, 104],
                "Product": [
                    "Credit card",
                    "Personal loan",
                    "Savings account",
                    "Money transfer",
                ],
                "Consumer complaint narrative": [
                    "XXXX Credit card late fees charged even though payment was sent early.",
                    "I am writing to file a complaint regarding my Personal loan interest adjustment.",
                    "Savings account withdrawals blocked without clear notice.",
                    "Money transfer processing delayed between regional branches.",
                ],
            }
        )
        mock_df.to_csv(raw_data_path, index=False)

    execute_preprocessing_pipeline(raw_data_path, processed_data_path)
