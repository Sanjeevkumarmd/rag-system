# RAG System — Retrieval-Augmented Generation Demo

A minimal but production-shaped RAG pipeline: chunk documents → embed locally →
store in FAISS → retrieve top-k → generate a grounded answer with an LLM.

## Architecture

```
data/*.txt, *.pdf
      │
      ▼
  ingest.py  ──►  chunk (500 words, 50-word overlap)
      │              │
      │              ▼
      │        sentence-transformers (all-MiniLM-L6-v2, local, free)
      │              │
      │              ▼
      └──────►  FAISS index (cosine similarity via normalized inner product)

  rag.py     ──►  embed query → retrieve top-k chunks → build prompt →
                   Claude generates answer grounded in retrieved context

  app.py     ──►  Streamlit UI wrapping rag.py for a live demo
```

## Setup (do this first)

```bash
pip install -r requirements.txt
```

This project uses a **free** LLM backend by default — no paid API key needed.
Pick one:

### Option A: Ollama (fully local, no signup, no key)
1. Install Ollama: https://ollama.com (Windows/Mac/Linux, works on a normal laptop)
2. Pull a small model:
   ```bash
   ollama pull llama3.2:3b
   ```
3. That's it — `LLM_BACKEND` defaults to `ollama`, no further setup needed.
   (Needs ~4-8GB free RAM. If your laptop struggles, try `ollama pull phi3` instead,
   it's smaller.)

### Option B: Groq (free hosted API, no credit card required)
1. Get a free key at https://console.groq.com
2. ```bash
   export LLM_BACKEND=groq
   export GROQ_API_KEY=your_key_here
   ```
   Groq's free tier is generous and the models (Llama 3.1 8B, etc.) are solid
   for a demo project, and responses are very fast.

Use whichever fits your setup — the rest of the pipeline (embeddings, FAISS,
retrieval) is identical either way, since embeddings run locally and for free
regardless of which LLM backend you pick.

## Usage

1. Drop your own `.txt` or `.pdf` files into `data/` (a sample doc about RAG itself
   is already there so you can test immediately).

2. Build the index:
   ```bash
   python ingest.py
   ```

3. Query from the command line:
   ```bash
   python rag.py
   ```
   or launch the UI:
   ```bash
   streamlit run app.py
   ```

## Design decisions worth knowing for an interview

- **Local embeddings (`all-MiniLM-L6-v2`)**: fast, free, no API cost per document.
  Good baseline; swap for `bge-large` or an API-based embedding model for higher
  retrieval quality at the cost of speed/money.
- **Chunking with overlap**: prevents an answer from being split across two chunks
  with neither one containing the full context.
- **FAISS `IndexFlatIP` with normalized vectors**: exact cosine-similarity search.
  Fine up to ~1M vectors; beyond that you'd want an approximate index (HNSW, IVF)
  to trade a little accuracy for speed.
- **Grounded generation**: the system prompt explicitly tells the model to say
  "I don't know" if the retrieved context doesn't answer the question — this is
  the main lever against hallucination in a RAG system.
- **Source citation**: every answer returns which document chunks were actually
  used, so the output is auditable, not a black box.

## Natural extensions (mention these even if you don't build them — shows you know
the roadmap)

- **Re-ranking**: retrieve top-20 with FAISS, then re-rank with a cross-encoder
  (e.g. `bge-reranker`) to get a more precise top-4.
- **Hybrid search**: combine dense vector search with BM25 keyword search for
  queries with exact terms/names/numbers that embeddings can miss.
- **Evaluation**: build a small labeled Q&A set and measure retrieval precision/
  recall and answer faithfulness (e.g. with RAGAS).
- **Chunking strategy**: move from fixed word-count chunks to semantic or
  structure-aware chunking (respecting headings, tables, code blocks).
- **Streaming responses** in the UI for better perceived latency.

## What this project demonstrates in an interview

Talk through the trade-offs above out loud — that's what separates "I followed a
tutorial" from "I understand RAG." Be ready to explain: why FAISS over a plain
list + cosine loop, why overlap matters, what happens if retrieval returns
irrelevant chunks, and how you'd scale this to millions of documents.
