# 🤖 AI Document Assistant

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Topic](https://img.shields.io/badge/Focus-RAG_%7C_NLP_%7C_Vector_Search-blue)](#)
[![Tests](https://img.shields.io/badge/Tests-Passing-success?style=flat&logo=githubactions&logoColor=white)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent Retrieval-Augmented Generation (**RAG**) document assistant built in Python. It parses domain documents (PDFs, Markdown, & raw text), splits them into semantic overlapping chunks, indexes them into a high-dimensional vector space, and accurately retrieves grounded context to answer user queries with verifiable citations.

---

## 🏛️ Architecture Overview

```
 [Raw Documents] (PDF / Markdown / TXT)
        │
        ▼
 [Document Loader & Chunker] ───► Sliding-window semantic chunking with overlap
        │
        ▼
 [Vector Index & TF-IDF Embeddings] ───► In-memory vector store & index serialization
        │
        ▼
 [User Query] ───► [Cosine Similarity Retrieval] ───► [Top-K Context] ───► [Answer Synthesis]
```

---

## ✨ Features

- **Multi-Format Document Processing**: Supports `.txt`, `.md`, and `.pdf` files.
- **Directory Ingestion**: Batch load entire document repositories via `load_directory()`.
- **Semantic Sliding Chunker**: Splits long documents with customizable token overlap to preserve context across boundaries.
- **In-Memory Vector Search**: Fast cosine similarity retrieval without mandatory third-party cloud API keys or vendor lock-in.
- **Index Persistence**: Export and reload pre-computed vector indices to/from JSON (`save_index` / `load_index`).
- **Interactive CLI & Automated Tests**: Real-time terminal interface and comprehensive `unittest` test suite.

---

## 🚀 Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/syedaamish11/ai-document-assistant.git
cd ai-document-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the interactive assistant
```bash
python main.py
```

### 4. Run automated test suite
```bash
python -m unittest discover -s tests
```

---

## 📁 Project Structure

```
ai-document-assistant/
├── data/
│   └── sample_knowledge.txt    # Default knowledge base for testing
├── tests/
│   └── test_rag.py             # Unit test suite (chunking, search, persistence)
├── document_loader.py          # PDF / Markdown / Text extraction & chunking logic
├── vector_store.py             # TF-IDF & Cosine Similarity search engine + persistence
├── main.py                     # CLI interaction loop & query answering
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack
- **Language**: Python 3.9+
- **Techniques**: Retrieval-Augmented Generation (RAG), Vector Embeddings, Cosine Similarity, TF-IDF
- **Libraries**: `pypdf`, `numpy`, `python-dotenv`, `rich`

---

## 👤 Author
- **Syed Aamish** — [GitHub Profile](https://github.com/syedaamish11)