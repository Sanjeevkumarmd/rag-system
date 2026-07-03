"""
ingest.py — Loads documents, chunks them, embeds them, and builds a FAISS index.

Run this once whenever your source documents change:
    python ingest.py --data_dir ./data --index_dir ./index
"""

import os
import re
import argparse
import pickle
from io import BytesIO
from pathlib import Path

import numpy as np
import faiss
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"}


def load_documents(data_dir: str) -> list[dict]:
    """Load .txt and .pdf files from data_dir. Returns list of {text, source}."""
    docs = []
    for path in Path(data_dir).glob("**/*"):
        if path.suffix.lower() == ".txt" and path.name != "urls.txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
            docs.append({"text": text, "source": str(path)})
        elif path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            docs.append({"text": text, "source": str(path)})
    return docs


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """Extract text from an in-memory uploaded file (.txt or .pdf), no disk write."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file_bytes.decode("utf-8", errors="ignore")


def fetch_url(url: str, timeout: int = 15) -> dict | None:
    """Download a web page and extract its readable text. Returns None on failure."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [skip] {url} -> {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text).strip()

    if len(text) < 100:
        print(f"  [skip] {url} -> page had too little extractable text")
        return None

    return {"text": text, "source": url}


def load_urls(data_dir: str) -> list[dict]:
    urls_file = Path(data_dir) / "urls.txt"
    if not urls_file.exists():
        return []

    urls = [
        line.strip()
        for line in urls_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not urls:
        return []

    print(f"Fetching {len(urls)} URL(s) from urls.txt ...")
    docs = []
    for url in urls:
        doc = fetch_url(url)
        if doc:
            docs.append(doc)
    return docs


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Splits text into overlapping word-based chunks."""
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


def build_index_from_docs(docs: list[dict], model: SentenceTransformer):
    """
    Build a FAISS index fully in memory from a list of {text, source} docs,
    using an already-loaded embedding model. Nothing touches disk.
    Returns (index, chunks, metadata) — or (None, [], []) if docs produced no chunks.
    """
    all_chunks = []
    chunk_metadata = []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        all_chunks.extend(chunks)
        chunk_metadata.extend([{"source": doc["source"]}] * len(chunks))

    if not all_chunks:
        return None, [], []

    embeddings = model.encode(all_chunks, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, all_chunks, chunk_metadata


def build_index(data_dir: str, index_dir: str, model_name: str = "all-MiniLM-L6-v2"):
    os.makedirs(index_dir, exist_ok=True)

    print(f"Loading documents from {data_dir} ...")
    docs = load_documents(data_dir)
    docs.extend(load_urls(data_dir))
    if not docs:
        print("No .txt/.pdf files or urls.txt entries found in data_dir. Add some and re-run.")
        return

    print(f"Loaded {len(docs)} document(s). Chunking ...")
    print(f"Loading embedding model ({model_name}) ...")
    model = SentenceTransformer(model_name)

    print("Embedding chunks ...")
    index, all_chunks, chunk_metadata = build_index_from_docs(docs, model)
    if index is None:
        print("No chunks produced. Aborting.")
        return

    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "chunks.pkl"), "wb") as f:
        pickle.dump({"chunks": all_chunks, "metadata": chunk_metadata}, f)

    print(f"Done. Index saved to {index_dir}/index.faiss ({len(all_chunks)} vectors)")


def add_url_to_file(url: str, data_dir: str = "./data") -> None:
    """Appends a URL to data_dir/urls.txt (creates the file if needed), skipping dupes."""
    urls_file = Path(data_dir) / "urls.txt"
    os.makedirs(data_dir, exist_ok=True)
    existing = set()
    if urls_file.exists():
        existing = {l.strip() for l in urls_file.read_text(encoding="utf-8").splitlines()}
    if url not in existing:
        with open(urls_file, "a", encoding="utf-8") as f:
            f.write(url + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--index_dir", default="./index")
    parser.add_argument("--model_name", default="all-MiniLM-L6-v2")
    parser.add_argument("--add_url", default=None, help="Add a URL to data/urls.txt before building.")
    args = parser.parse_args()

    if args.add_url:
        add_url_to_file(args.add_url, args.data_dir)
        print(f"Added {args.add_url} to {args.data_dir}/urls.txt")

    build_index(args.data_dir, args.index_dir, args.model_name)