"""
app.py — Project root entry point for the CrediTrust RAG Complaint Chatbot.

This thin wrapper imports and launches the Gradio app defined in src/app.py.
Run: python app.py
"""

import os
import sys

# Add src/ to the Python path so imports work from both root and src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from app import create_app, load_vector_store, _get_llm_pipeline

if __name__ == "__main__":
    print("[CrediTrust RAG] Starting Intelligent Complaint Analysis Portal...")
    print("[CrediTrust RAG] Pre-loading vector store and language model...")

    try:
        load_vector_store()
        _get_llm_pipeline()
        print("[CrediTrust RAG] ✓ All resources loaded successfully.")
    except Exception as e:
        print(f"[CrediTrust RAG] ⚠ Warning during pre-load: {e}")
        print("[CrediTrust RAG] Resources will be loaded on first query.")

    demo = create_app()
    print("[CrediTrust RAG] ✓ Launching on http://127.0.0.1:7860")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
