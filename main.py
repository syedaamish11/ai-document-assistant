"""
AI Document Assistant - Main Entrypoint
CLI application demonstrating Document Ingestion, Chunking, Vector Indexing, and Query Answering.
"""

import os
import sys
from document_loader import DocumentLoader
from vector_store import SimpleVectorStore

def main():
    print("=" * 60)
    print("   AI Document Assistant (RAG Pipeline Demo)")
    print("=" * 60)

    loader = DocumentLoader(chunk_size=120, chunk_overlap=30)
    store = SimpleVectorStore()

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    sample_file = os.path.join(data_dir, "sample_knowledge.txt")

    if not os.path.exists(sample_file):
        print(f"[Error] Knowledge file not found at {sample_file}")
        return

    print(f"\n[+] Loading document: {os.path.basename(sample_file)}...")
    raw_text = loader.load_text(sample_file)
    chunks = loader.chunk_text(raw_text, source_name="sample_knowledge.txt")
    print(f"[+] Document divided into {len(chunks)} semantic chunks.")

    print("[+] Building vector index and TF-IDF weights...")
    store.add_documents(chunks)
    print("[+] Vector store ready for retrieval queries!\n")

    print("-" * 60)
    print("Ask any question based on the document (or type 'exit' to quit):")
    print("Example queries:")
    print("  - What are the core components of RAG?")
    print("  - Why do LLMs hallucinate?")
    print("-" * 60)

    while True:
        try:
            query = input("\n[Question] > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Exiting AI Document Assistant. Happy coding!")
                break

            matches = store.search(query, top_k=2)
            if not matches:
                print("No relevant context found in documents.")
                continue

            print(f"\n--- Retrieved Context ({len(matches)} chunks) ---")
            for idx, (doc, score) in enumerate(matches, 1):
                print(f"\n[Match #{idx}] (Relevance Score: {score:.3f} | Source: {doc['source']})")
                print(f"\"{doc['text']}\"")

            print("\n[Synthesized Answer]:")
            print(f"Based on the retrieved context, {matches[0]['text']}")

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

if __name__ == "__main__":
    main()
