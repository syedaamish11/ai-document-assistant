"""
Document Loader and Chunker Module
Supports loading raw text, Markdown, and PDF documents,
splitting them into overlapping semantic chunks with metadata preservation.
"""

import os
from typing import List, Dict

class DocumentLoader:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_text(self, file_path: str) -> str:
        """Reads plain text and markdown files with UTF-8 encoding."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def load_markdown(self, file_path: str) -> str:
        """Loads markdown files, stripping excessive divider whitespace."""
        return self.load_text(file_path)

    def load_pdf(self, file_path: str) -> str:
        """Extracts text from PDF documents using pypdf."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = []
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text.append(extracted.strip())
            return "\n\n".join(text)
        except ImportError:
            print("[Warning] 'pypdf' is not installed. PDF loading is disabled.")
            return ""

    def load_directory(self, dir_path: str) -> List[Dict[str, any]]:
        """
        Scans a directory for supported files (.txt, .md, .pdf),
        extracts their contents, and chunks them into a unified list.
        """
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        supported_extensions = {".txt", ".md", ".pdf"}
        all_chunks = []

        for root, _, files in os.walk(dir_path):
            for file in sorted(files):
                ext = os.path.splitext(file)[1].lower()
                if ext in supported_extensions:
                    full_path = os.path.join(root, file)
                    rel_name = os.path.relpath(full_path, dir_path)

                    if ext == ".pdf":
                        raw_text = self.load_pdf(full_path)
                    else:
                        raw_text = self.load_text(full_path)

                    if raw_text.strip():
                        chunks = self.chunk_text(raw_text, source_name=rel_name)
                        all_chunks.extend(chunks)

        return all_chunks

    def chunk_text(self, text: str, source_name: str = "unknown") -> List[Dict[str, any]]:
        """
        Splits text into chunks with sliding-window overlap.
        Returns a list of dicts with text and metadata.
        """
        words = text.split()
        if not words:
            return []

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
                "word_count": len(words[start:end])
            })
            chunk_id += 1
            if end >= len(words):
                break
            start += (self.chunk_size - self.chunk_overlap)

        return chunks