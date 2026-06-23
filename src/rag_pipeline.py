"""
src/rag_pipeline.py
-------------------
Core RAG pipeline for CrediTrust Financial Complaint Analysis.

Architecture:
  1. RETRIEVER  — embeds user query, runs FAISS similarity search
  2. PROMPT     — formats analyst-style prompt with retrieved context
  3. GENERATOR  — feeds prompt to a local HuggingFace LLM (flan-t5-base)
  4. run_rag()  — end-to-end function used by the Gradio UI

LLM backends (priority order):
  - GROQ_API_KEY set  → Groq API (llama3-8b-8192, fast)
  - OPENAI_API_KEY set → OpenAI API (gpt-3.5-turbo)
  - Default           → HuggingFace local (google/flan-t5-base, no API key)
"""

import os
import re
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ─── Configuration ────────────────────────────────────────────────────────────

VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "vector_store")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5

ANALYST_PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust Financial, \
a digital finance company operating across East Africa. Your job is to analyze customer \
complaint data and provide clear, actionable insights to internal teams.

Use ONLY the retrieved complaint excerpts below to formulate your answer. \
Be specific and cite patterns you observe. \
If the context does not contain enough information to answer confidently, \
clearly state: "I don't have enough information in the available complaints to answer this fully."

--- Retrieved Complaint Excerpts ---
{context}
---

Question: {question}

Analyst Answer:"""


# ─── Singleton resources (loaded once at module import) ───────────────────────

_embedding_model: Optional[HuggingFaceEmbeddings] = None
_vector_store: Optional[FAISS] = None
_llm_pipeline = None


def _get_embedding_model() -> HuggingFaceEmbeddings:
    """Load and cache the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        print(f"[RAG] Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model


def load_vector_store(path: str = VECTOR_STORE_PATH) -> FAISS:
    """Load and cache the FAISS vector store from disk."""
    global _vector_store
    if _vector_store is None:
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(
                f"Vector store not found at: {abs_path}\n"
                "Run 'python src/vector_indexing.py' first to build the index."
            )
        print(f"[RAG] Loading FAISS vector store from: {abs_path}")
        _vector_store = FAISS.load_local(
            abs_path,
            _get_embedding_model(),
            allow_dangerous_deserialization=True,
        )
        print(f"[RAG] Vector store loaded successfully.")
    return _vector_store


def _get_llm_pipeline():
    """
    Load and cache the LLM. Checks for API keys in order:
      1. GROQ_API_KEY  → Groq (llama3-8b-8192)
      2. OPENAI_API_KEY → OpenAI (gpt-3.5-turbo)
      3. Default        → HuggingFace local (google/flan-t5-base)
    """
    global _llm_pipeline

    if _llm_pipeline is not None:
        return _llm_pipeline

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if groq_key:
        print("[RAG] Using Groq API (llama3-8b-8192)")
        try:
            from groq import Groq

            client = Groq(api_key=groq_key)

            def groq_generate(prompt: str) -> str:
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=0.3,
                )
                return response.choices[0].message.content.strip()

            _llm_pipeline = groq_generate
            return _llm_pipeline
        except Exception as e:
            print(f"[RAG] Groq init failed: {e}. Falling back to local model.")

    if openai_key:
        print("[RAG] Using OpenAI API (gpt-3.5-turbo)")
        try:
            from openai import OpenAI

            client = OpenAI(api_key=openai_key)

            def openai_generate(prompt: str) -> str:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=0.3,
                )
                return response.choices[0].message.content.strip()

            _llm_pipeline = openai_generate
            return _llm_pipeline
        except Exception as e:
            print(f"[RAG] OpenAI init failed: {e}. Falling back to local model.")

    # Default: HuggingFace local flan-t5-base
    print("[RAG] Loading local HuggingFace model: google/flan-t5-base")
    try:
        from transformers import pipeline

        pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=300,
            do_sample=False,
            device=-1,  # CPU
        )

        def hf_generate(prompt: str) -> str:
            # flan-t5 works best with a condensed input; trim if too long
            trimmed = prompt[:2048]
            result = pipe(trimmed)
            return result[0]["generated_text"].strip()

        _llm_pipeline = hf_generate
        print("[RAG] Local flan-t5-base model ready.")
        return _llm_pipeline
    except Exception as e:
        raise RuntimeError(f"[RAG] Failed to load any LLM: {e}")


# ─── Core RAG Functions ────────────────────────────────────────────────────────


def retrieve(query: str, k: int = DEFAULT_TOP_K, product_filter: Optional[str] = None):
    """
    Embed the query and retrieve top-k similar complaint chunks from FAISS.

    Args:
        query: The user's natural-language question.
        k: Number of chunks to retrieve.
        product_filter: Optional product category to filter results
                        (e.g. "Credit card", "Personal loan").

    Returns:
        List of LangChain Document objects with page_content and metadata.
    """
    vs = load_vector_store()
    docs = vs.similarity_search(query, k=k * 2 if product_filter else k)

    if product_filter and product_filter.lower() != "all":
        docs = [
            d for d in docs
            if d.metadata.get("product", "").lower() == product_filter.lower()
        ]

    return docs[:k]


def build_prompt(question: str, context_docs: list) -> str:
    """
    Format the analyst prompt with retrieved context.

    Args:
        question: The user's question.
        context_docs: List of LangChain Document objects.

    Returns:
        Formatted prompt string ready for the LLM.
    """
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        meta = doc.metadata
        complaint_id = meta.get("complaint_id", "N/A")
        product = meta.get("product", "N/A")
        issue = meta.get("issue", "")
        chunk_label = f"[Excerpt {i} | ID: {complaint_id} | Product: {product}"
        if issue:
            chunk_label += f" | Issue: {issue}"
        chunk_label += "]"
        context_parts.append(f"{chunk_label}\n{doc.page_content}")

    context_str = "\n\n".join(context_parts)
    return ANALYST_PROMPT_TEMPLATE.format(context=context_str, question=question)


def generate_answer(prompt: str) -> str:
    """
    Send the formatted prompt to the LLM and return the generated response.

    Args:
        prompt: Fully formatted prompt string.

    Returns:
        LLM-generated answer as a string.
    """
    llm = _get_llm_pipeline()
    return llm(prompt)


def run_rag(
    question: str,
    k: int = DEFAULT_TOP_K,
    product_filter: Optional[str] = None,
) -> dict:
    """
    End-to-end RAG pipeline: retrieve → prompt → generate.

    Args:
        question: The user's natural-language question.
        k: Number of complaint chunks to retrieve.
        product_filter: Optional product category filter.

    Returns:
        Dict with keys:
          - "answer": str — LLM-generated answer
          - "sources": list of dicts — metadata for each retrieved chunk
          - "context_docs": list — raw Document objects (for UI display)
    """
    if not question or not question.strip():
        return {
            "answer": "Please enter a question to analyze complaint data.",
            "sources": [],
            "context_docs": [],
        }

    # Step 1: Retrieve
    context_docs = retrieve(question, k=k, product_filter=product_filter)

    if not context_docs:
        return {
            "answer": (
                "No relevant complaint records were found for your query"
                + (f" in the '{product_filter}' category" if product_filter else "")
                + ". Try rephrasing or selecting a different product filter."
            ),
            "sources": [],
            "context_docs": [],
        }

    # Step 2: Build prompt
    prompt = build_prompt(question, context_docs)

    # Step 3: Generate
    answer = generate_answer(prompt)

    # Step 4: Format sources for display
    sources = []
    for doc in context_docs:
        meta = doc.metadata
        sources.append(
            {
                "complaint_id": meta.get("complaint_id", "N/A"),
                "product": meta.get("product", "N/A"),
                "issue": meta.get("issue", ""),
                "chunk_index": meta.get("chunk_sequence", meta.get("chunk_index", 0)),
                "excerpt": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
            }
        )

    return {
        "answer": answer,
        "sources": sources,
        "context_docs": context_docs,
    }


# ─── Standalone Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_questions = [
        "What are the most common complaints about credit cards?",
        "Why are customers unhappy with money transfers?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"QUESTION: {q}")
        print("=" * 60)
        result = run_rag(q, k=3)
        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nSOURCES ({len(result['sources'])}):")
        for s in result["sources"]:
            print(f"  - [{s['complaint_id']}] {s['product']} | {s['issue']}")
            print(f"    Excerpt: {s['excerpt'][:100]}...")
