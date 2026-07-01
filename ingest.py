"""
ingest.py — Loads documents, chunks them, embeds them, and builds a FAISS index.

Run this once whenever your source documents change:
    python ingest.py --data_dir ./data --index_dir ./index
"""

import os
import argparse
import pickle
from pathlib import Path

import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def load_documents(data_dir: str) -> list[dict]:
    """Load .txt and .pdf files from data_dir. Returns list of {text, source}."""
    docs = []
    for path in Path(data_dir).glob("**/*"):
        if path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
            docs.append({"text": text, "source": str(path)})
        elif path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            docs.append({"text": text, "source": str(path)})
    return docs


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits text into overlapping word-based chunks.
    Overlap matters: it stops answers from being cut in half at a chunk boundary.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_index(data_dir: str, index_dir: str, model_name: str = "all-MiniLM-L6-v2"):
    os.makedirs(index_dir, exist_ok=True)

    print(f"Loading documents from {data_dir} ...")
    docs = load_documents(data_dir)
    if not docs:
        print("No .txt or .pdf files found in data_dir. Add some and re-run.")
        return

    print(f"Loaded {len(docs)} document(s). Chunking ...")
    all_chunks = []
    chunk_metadata = []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        all_chunks.extend(chunks)
        chunk_metadata.extend([{"source": doc["source"]}] * len(chunks))

    print(f"Created {len(all_chunks)} chunks. Loading embedding model ({model_name}) ...")
    model = SentenceTransformer(model_name)

    print("Embedding chunks ...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "chunks.pkl"), "wb") as f:
        pickle.dump({"chunks": all_chunks, "metadata": chunk_metadata}, f)

    print(f"Done. Index saved to {index_dir}/index.faiss ({len(all_chunks)} vectors, dim={dim})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--index_dir", default="./index")
    parser.add_argument("--model_name", default="all-MiniLM-L6-v2")
    args = parser.parse_args()
    build_index(args.data_dir, args.index_dir, args.model_name)
