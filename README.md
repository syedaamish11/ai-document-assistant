# 🧠 AI Document Assistant

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Topic](https://img.shields.io/badge/Focus-RAG_%7C_NLP_%7C_Vector_Search-blue)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent Retrieval-Augmented Generation (**RAG**) document assistant built in Python. It parses domain documents (PDFs & raw text), splits them into semantic overlapping chunks, indexes them into a vector space, and accurately retrieves grounded context to answer user queries with verifiable citations.

---

## 📌 Architecture Overview

```
 [Raw Documents] (PDF / TXT)
        │
        ▼
 [Document Loader & Chunker] ─── Sliding-window semantic chunking
        │
        ▼
 [Vector Index / TF-IDF Embeddings] ─── High-dimensional vector space
        │
        ▼
 [User Query] ──► [Cosine Similarity Retrieval] ──► [Top-K Context] ──► [Answer Synthesis]
```

---

## ✨ Features

- **Document Processing**: Supports both `.txt` and `.pdf` files.
- **Semantic Sliding Chunker**: Splits long documents with customizable token overlap to preserve paragraph context.
- **In-Memory Vector Search**: Fast cosine similarity retrieval without mandatory cloud API dependencies.
- **Extensible LLM Integration**: Ready to hook into Gemini API, OpenAI, or local Ollama models.
- **Interactive CLI**: Fast terminal interface to test retrieval quality in real-time.

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

---

## 📂 Project Structure

```
ai-document-assistant/
├── data/
│   └── sample_knowledge.txt    # Default knowledge base for testing
├── document_loader.py          # PDF / Text extraction & chunking logic
├── vector_store.py             # TF-IDF & Cosine Similarity search engine
├── main.py                     # CLI interaction loop
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack
- **Language**: Python 3
- **Techniques**: Retrieval-Augmented Generation (RAG), Vector Embeddings, Cosine Similarity, TF-IDF
- **Libraries**: `pypdf`, `numpy`, `python-dotenv`

---

## 👤 Author
- **Syed Aamish** — [GitHub Profile](https://github.com/syedaamish11)
