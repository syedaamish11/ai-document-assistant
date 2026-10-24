"""
Document Loader and Chunker Module
Supports loading raw text and PDF documents, splitting them into overlapping semantic chunks.
"""

import os
from typing import List, Dict

class DocumentLoader:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_text(self, file_path: str) -> str:
        """Reads plain text files."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def load_pdf(self, file_path: str) -> str:
        """Extracts text from PDF documents using pypdf."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = []
            for page_num, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted:
                    text.append(extracted)
            return "\n\n".join(text)
        except ImportError:
            print("[Warning] 'pypdf' is not installed. PDF loading is disabled.")
            return ""

    def chunk_text(self, text: str, source_name: str = "unknown") -> List[Dict[str, any]]:
        """
        Splits text into chunks with sliding-window overlap.
        Returns a list of dicts with text and metadata.
        """
        words = text.split()
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_content = " ".join(words[start:end])
            chunks.append({
                "id": chunk_id,
                "source": source_name,
                "text": chunk_content,
                "word_count": len(chunk_content.split())
            })
            chunk_id += 1
            if end >= len(words):
                break
            start += (self.chunk_size - self.chunk_overlap)

        return chunks
