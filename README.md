# 🏦 CrediTrust Financial — Intelligent Complaint Analysis Portal

> A RAG-powered chatbot that transforms unstructured CFPB customer complaint data into actionable insights for financial services teams.

[![Unit Tests](https://github.com/Guyatu1627/rag-complaint-chatbot/actions/workflows/unittests.yml/badge.svg)](https://github.com/Guyatu1627/rag-complaint-chatbot/actions/workflows/unittests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Project Overview

**CrediTrust Financial** serves over 500,000 customers across East Africa through a mobile-first platform offering credit cards, personal loans, savings accounts, and money transfers. This tool gives Product Managers, Compliance Officers, and Support teams the ability to ask plain-English questions about customer complaint patterns — and receive AI-generated, evidence-backed answers in seconds.

### Business KPIs
| KPI | Before | After |
|-----|--------|-------|
| Time to identify complaint trend | Days | **Minutes** |
| Analyst required for complaint queries | Yes | **No** |
| Issue detection | Reactive | **Proactive** |

---

## 🏗️ Architecture

```
User Question
     │
     ▼
[Embedding Model] (all-MiniLM-L6-v2, 384 dims)
     │
     ▼
[FAISS Vector Store] ──→ Top-5 Relevant Complaint Chunks
     │
     ▼
[Prompt Engineering] ──→ Analyst-style context injection
     │
     ▼
[LLM Generator] (google/flan-t5-base or Groq/OpenAI via env var)
     │
     ▼
[Gradio UI] ──→ Answer + Source Excerpts displayed to user
```

---

## 📁 Project Structure

```
rag-complaint-chatbot/
├── .github/
│   └── workflows/
│       └── unittests.yml          # CI/CD: pytest on push/PR
├── data/
│   ├── raw/
│   │   └── complaints.csv         # Full CFPB dataset (~6 GB)
│   └── processed/
│       └── filtered_complaints.csv # Cleaned 4-product subset
├── notebooks/
│   ├── task1_eda.ipynb            # Task 1: EDA & Preprocessing
│   └── task3_evaluation.ipynb     # Task 3: RAG Evaluation Table
├── src/
│   ├── eda_preprocessing.py       # Text cleaning pipeline
│   ├── vector_indexing.py         # Chunking, embedding, FAISS indexing
│   ├── rag_pipeline.py            # Core RAG logic (retrieve + generate)
│   └── app.py                     # Gradio UI
├── tests/
│   └── test_rag_pipeline.py       # Unit tests
├── vector_store/
│   ├── index.faiss                # FAISS binary index (~51 MB)
│   └── index.pkl                  # Docstore with metadata
├── app.py                         # Project root entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Guyatu1627/rag-complaint-chatbot.git
cd rag-complaint-chatbot
pip install -r requirements.txt
```

### 2. Prepare Data (if vector store not yet built)

```bash
# Step 1: Preprocess raw data
python src/eda_preprocessing.py

# Step 2: Build FAISS vector store (~5–15 min depending on hardware)
python src/vector_indexing.py
```

> **Note:** A pre-built `vector_store/` is included in this repo, so steps 1–2 are only needed if you want to rebuild from scratch.

### 3. Launch the Chatbot

```bash
python app.py
```

Open your browser at **http://127.0.0.1:7860**

### 4. Optional: Use a Faster LLM

```bash
# Use Groq API (fast, free tier)
$env:GROQ_API_KEY = "your-groq-key-here"
python app.py

# Use OpenAI
$env:OPENAI_API_KEY = "your-openai-key-here"
python app.py
```

---

## 🔬 Technical Details

### Task 1: EDA & Preprocessing

**Dataset:** CFPB Consumer Complaint Database (464K+ records across all financial products)

**Filtering:**
- Retained: `Credit card`, `Personal loan`, `Savings account`, `Money transfer`
- Dropped: Records with no consumer narrative

**Text Cleaning:**
- Lowercased all text
- Removed `XXXX` anonymization placeholders
- Stripped boilerplate phrases (`"I am writing to file a complaint..."`)
- Normalized whitespace and special characters

**Key EDA Findings:**
- Credit Cards dominate complaint volume (~60–70%)
- Narrative length ranges from <10 words to 1,500+ words (high variance)
- Median narrative: ~100–150 words

### Task 2: Chunking & Embedding

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk size | 500 chars | Balances semantic completeness vs. vector precision |
| Chunk overlap | 50 chars | Preserves context across chunk boundaries |
| Splitter | `RecursiveCharacterTextSplitter` | Respects sentence/paragraph boundaries |
| Sample size | 12,000 complaints | Stratified by product for representation |
| Embedding model | `all-MiniLM-L6-v2` | Fast (384d), high quality for semantic similarity |
| Vector DB | FAISS | Local, no server required, fast similarity search |

### Task 3: RAG Pipeline

**Retriever:** FAISS cosine similarity search, top-k=5

**Prompt Template:**
```
You are a financial analyst assistant for CrediTrust Financial...
Use ONLY the retrieved complaint excerpts below to formulate your answer.
Context: {retrieved_chunks}
Question: {user_question}
Analyst Answer:
```

**Generator (default):** `google/flan-t5-base` — local CPU inference, no API key
**Generator (optional):** Groq (llama3-8b-8192) or OpenAI (gpt-3.5-turbo) via env vars

### Task 4: Gradio UI

Features:
- 💬 Plain-English question input
- 📦 Product category filter (All / Credit Card / Personal Loan / Savings Account / Money Transfer)
- 🤖 AI-generated analyst answer
- 📋 Source chunk display with complaint ID, product, and excerpt
- 🗑️ Clear button to reset conversation
- 💡 Example questions for guided exploration

---

## 📊 RAG Evaluation Results

| ID | Question | Quality Score | Comments |
|----|----------|:---:|---------|
| Q1 | Credit card billing and fees | 4/5 | Good coverage, could be more specific |
| Q2 | Money transfer frustrations | 4/5 | Correctly identifies delays and fees |
| Q3 | Savings account access problems | 3/5 | Partially addresses root causes |
| Q4 | Personal loan repayment issues | 4/5 | Well-grounded in retrieved context |
| Q5 | Fraud across all products | 3/5 | Broad but not deeply analyzed |
| Q6 | Customer service experience | 4/5 | Strong pattern identification |
| Q7 | Credit card duplicate charges | 5/5 | Excellent specificity with evidence |
| Q8 | Loan modification/forbearance | 3/5 | Limited index coverage for topic |

**Average Quality Score: 3.75 / 5.0**

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥2.0.1 | Data processing |
| `sentence-transformers` | ≥2.2.2 | Embedding model |
| `faiss-cpu` | ≥1.7.4 | Vector similarity search |
| `langchain` | ≥0.1.0 | RAG pipeline orchestration |
| `transformers` | ≥4.37.0 | HuggingFace LLM |
| `torch` | ≥2.0.0 | ML framework |
| `gradio` | ≥4.12.0 | Web UI |
| `pytest` | ≥7.4.0 | Testing |

---

## 👤 Author

**Guyatu** — 10 Academy Week 7 Challenge  
*Intelligent Complaint Analysis for Financial Services*

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.