"""
Unit tests for AI Document Assistant RAG Pipeline
Tests chunking, vector embeddings, similarity search, and index persistence.
"""

import os
import sys
import unittest
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from document_loader import DocumentLoader
from vector_store import SimpleVectorStore

class TestAIDocumentAssistant(unittest.TestCase):
    def setUp(self):
        self.loader = DocumentLoader(chunk_size=10, chunk_overlap=3)
        self.store = SimpleVectorStore()

    def test_chunking_overlap(self):
        sample_text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12 word13 word14 word15"
        chunks = self.loader.chunk_text(sample_text, source_name="test.txt")
        self.assertGreater(len(chunks), 1)
        # Check first chunk has 10 words
        self.assertEqual(chunks[0]["word_count"], 10)
        # Check metadata
        self.assertEqual(chunks[0]["source"], "test.txt")

    def test_vector_search_relevance(self):
        docs = [
            {"id": 0, "source": "ai.txt", "text": "Neural networks and transformers power modern large language models.", "word_count": 9},
            {"id": 1, "source": "cooking.txt", "text": "Baking sourdough bread requires flour water salt and wild yeast starter.", "word_count": 11},
            {"id": 2, "source": "ai.txt", "text": "Retrieval augmented generation connects LLMs to external vector databases.", "word_count": 9}
        ]
        self.store.add_documents(docs)
        stats = self.store.get_statistics()
        self.assertEqual(stats["total_chunks"], 3)
        self.assertIn("ai.txt", stats["unique_sources"])

        # Search for AI related term
        results = self.store.search("transformers neural language", top_k=1)
        self.assertEqual(len(results), 1)
        top_doc, score = results[0]
        self.assertEqual(top_doc["id"], 0)
        self.assertGreater(score, 0.0)

        # Search for cooking term
        results_cooking = self.store.search("sourdough yeast flour", top_k=1)
        self.assertEqual(results_cooking[0][0]["id"], 1)

    def test_index_save_and_load(self):
        docs = [
            {"id": 0, "source": "doc1.txt", "text": "Continuous integration and automated testing prevent software regressions.", "word_count": 9}
        ]
        self.store.add_documents(docs)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.store.save_index(tmp_path)
            new_store = SimpleVectorStore()
            new_store.load_index(tmp_path)

            results = new_store.search("automated testing regression", top_k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0][0]["id"], 0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    unittest.main()