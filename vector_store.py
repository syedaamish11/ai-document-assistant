"""
Vector Store and Similarity Search Engine
Implements an in-memory TF-IDF semantic vector space for fast retrieval without external API locks.
"""

import math
import re
from typing import List, Dict, Tuple

class SimpleVectorStore:
    def __init__(self):
        self.documents: List[Dict[str, any]] = []
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())

    def add_documents(self, chunks: List[Dict[str, any]]):
        """Indexes document chunks and computes TF-IDF weights."""
        self.documents = chunks
        doc_count = len(chunks)
        if doc_count == 0:
            return

        doc_frequencies: Dict[str, int] = {}

        # 1. Build term frequency and vocabulary
        for doc in self.documents:
            tokens = set(self._tokenize(doc["text"]))
            for token in tokens:
                doc_frequencies[token] = doc_frequencies.get(token, 0) + 1

        self.vocabulary = {term: idx for idx, term in enumerate(doc_frequencies.keys())}

        # 2. Compute Inverse Document Frequency (IDF)
        for term, freq in doc_frequencies.items():
            self.idf[term] = math.log((1 + doc_count) / (1 + freq)) + 1.0

        # 3. Compute vector representations for all chunks
        for doc in self.documents:
            doc["vector"] = self._compute_vector(doc["text"])

    def _compute_vector(self, text: str) -> Dict[str, float]:
        tokens = self._tokenize(text)
        if not tokens:
            return {}

        tf: Dict[str, float] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0.0) + 1.0

        vector: Dict[str, float] = {}
        norm_sq = 0.0
        for token, count in tf.items():
            if token in self.idf:
                weight = (count / len(tokens)) * self.idf[token]
                vector[token] = weight
                norm_sq += weight * weight

        # Normalize to unit vector
        norm = math.sqrt(norm_sq)
        if norm > 0:
            for token in vector:
                vector[token] /= norm

        return vector

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, any], float]]:
        """Performs cosine similarity search against indexed document chunks."""
        query_vector = self._compute_vector(query)
        if not query_vector:
            return []

        results = []
        for doc in self.documents:
            doc_vec = doc.get("vector", {})
            score = sum(query_vector[t] * doc_vec.get(t, 0.0) for t in query_vector)
            if score > 0.0:
                results.append((doc, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
