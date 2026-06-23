"""
tests/test_rag_pipeline.py
--------------------------
Unit tests for the CrediTrust RAG pipeline components.

Tests cover:
- Text cleaning (eda_preprocessing.py)
- Prompt building (rag_pipeline.py)
- RAG pipeline with mocked vector store (rag_pipeline.py)
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# ── Path Setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── Tests: Text Cleaning ───────────────────────────────────────────────────────

class TestCleanComplaintText(unittest.TestCase):
    """Unit tests for eda_preprocessing.clean_complaint_text()"""

    def setUp(self):
        from eda_preprocessing import clean_complaint_text
        self.clean = clean_complaint_text

    def test_removes_boilerplate_phrase(self):
        raw = "I am writing to file a complaint regarding my credit card."
        cleaned = self.clean(raw)
        self.assertNotIn("i am writing to file a complaint regarding", cleaned)

    def test_removes_xxxx_placeholders(self):
        raw = "My account XXXX was charged XXXX dollars."
        cleaned = self.clean(raw)
        self.assertNotIn("xxxx", cleaned)

    def test_lowercases_text(self):
        raw = "My Credit Card Late Fee Was Charged."
        cleaned = self.clean(raw)
        self.assertEqual(cleaned, cleaned.lower())

    def test_handles_non_string_input(self):
        cleaned = self.clean(None)
        self.assertEqual(cleaned, "")
        cleaned = self.clean(42)
        self.assertEqual(cleaned, "")

    def test_removes_special_characters(self):
        raw = "Dispute #1234 — amount: $500 @bank!"
        cleaned = self.clean(raw)
        self.assertNotIn("#", cleaned)
        self.assertNotIn("@", cleaned)
        self.assertNotIn("—", cleaned)

    def test_strips_extra_whitespace(self):
        raw = "my  credit   card    statement"
        cleaned = self.clean(raw)
        self.assertNotIn("  ", cleaned)

    def test_empty_string_input(self):
        cleaned = self.clean("")
        self.assertEqual(cleaned, "")

    def test_preserves_meaningful_content(self):
        raw = "Late payment fee was charged on my account without any prior notice."
        cleaned = self.clean(raw)
        self.assertIn("late", cleaned)
        self.assertIn("payment", cleaned)
        self.assertIn("fee", cleaned)


# ── Tests: Prompt Building ─────────────────────────────────────────────────────

class TestBuildPrompt(unittest.TestCase):
    """Unit tests for rag_pipeline.build_prompt()"""

    def setUp(self):
        from rag_pipeline import build_prompt
        self.build_prompt = build_prompt

        # Create mock documents
        doc1 = MagicMock()
        doc1.page_content = "Customer complained about late fee on credit card."
        doc1.metadata = {
            "complaint_id": "12345",
            "product": "Credit card",
            "issue": "Billing disputes",
        }

        doc2 = MagicMock()
        doc2.page_content = "Unauthorized charge appeared on statement."
        doc2.metadata = {
            "complaint_id": "67890",
            "product": "Credit card",
            "issue": "Fraud",
            "chunk_sequence": 0,
        }

        self.mock_docs = [doc1, doc2]

    def test_includes_question_in_prompt(self):
        question = "Why are customers unhappy with credit cards?"
        prompt = self.build_prompt(question, self.mock_docs)
        self.assertIn(question, prompt)

    def test_includes_complaint_id_in_context(self):
        prompt = self.build_prompt("test question", self.mock_docs)
        self.assertIn("12345", prompt)
        self.assertIn("67890", prompt)

    def test_includes_product_in_context(self):
        prompt = self.build_prompt("test question", self.mock_docs)
        self.assertIn("Credit card", prompt)

    def test_includes_document_content(self):
        prompt = self.build_prompt("test question", self.mock_docs)
        self.assertIn("late fee", prompt)
        self.assertIn("Unauthorized charge", prompt)

    def test_returns_string(self):
        prompt = self.build_prompt("test", self.mock_docs)
        self.assertIsInstance(prompt, str)

    def test_handles_empty_documents(self):
        prompt = self.build_prompt("test question", [])
        # Prompt should still be returned (though context will be empty)
        self.assertIsInstance(prompt, str)
        self.assertIn("test question", prompt)

    def test_analyst_persona_present(self):
        prompt = self.build_prompt("What is the main issue?", self.mock_docs)
        self.assertIn("CrediTrust", prompt)


# ── Tests: run_rag with mocked dependencies ────────────────────────────────────

class TestRunRag(unittest.TestCase):
    """Unit tests for rag_pipeline.run_rag() with mocked FAISS and LLM."""

    def setUp(self):
        # Create mock documents
        doc = MagicMock()
        doc.page_content = "Late fee was charged incorrectly on my credit card."
        doc.metadata = {
            "complaint_id": "99001",
            "product": "Credit card",
            "issue": "Billing disputes",
            "chunk_sequence": 0,
        }
        self.mock_docs = [doc]

        # Reset ALL module-level singletons before each test to prevent cross-test contamination
        import rag_pipeline
        rag_pipeline._vector_store = None
        rag_pipeline._llm_pipeline = None
        rag_pipeline._embedding_model = None

    def _setup_rag_module(self, docs, llm_response="Generated answer."):
        """Helper to inject mock vector store and LLM into rag_pipeline module."""
        import rag_pipeline
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = docs
        rag_pipeline._vector_store = mock_store
        rag_pipeline._llm_pipeline = lambda prompt: llm_response
        return mock_store

    def test_returns_dict_with_expected_keys(self):
        self._setup_rag_module(self.mock_docs, "Customers complain about billing errors.")
        import rag_pipeline
        result = rag_pipeline.run_rag("Why are people unhappy with credit cards?")
        self.assertIn("answer", result)
        self.assertIn("sources", result)
        self.assertIn("context_docs", result)

    def test_empty_question_returns_message(self):
        self._setup_rag_module(self.mock_docs)
        import rag_pipeline
        result = rag_pipeline.run_rag("")
        self.assertIn("Please enter a question", result["answer"])
        self.assertEqual(result["sources"], [])

    def test_no_docs_returns_not_found_message(self):
        self._setup_rag_module([])  # Empty docs list
        import rag_pipeline
        result = rag_pipeline.run_rag("obscure query with no matches")
        self.assertIn("No relevant complaint records", result["answer"])
        self.assertEqual(result["sources"], [])

    def test_sources_have_correct_structure(self):
        self._setup_rag_module(self.mock_docs)
        import rag_pipeline
        result = rag_pipeline.run_rag("credit card fee complaints")
        self.assertGreater(len(result["sources"]), 0)
        source = result["sources"][0]
        self.assertIn("complaint_id", source)
        self.assertIn("product", source)
        self.assertIn("excerpt", source)


# ── Test Runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
