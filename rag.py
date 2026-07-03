"""
rag.py — Core RAG pipeline + Groq general AI + streaming + key-point extraction.
Supports Professional Mode and Gen Z/Alpha Mode prompts with concise straight answers.
"""

import os
import json as _json
import pickle
import re

import faiss
import requests
from sentence_transformers import SentenceTransformer

LLM_BACKEND = os.environ.get("LLM_BACKEND", "groq")
GROQ_MODEL  = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"


def _get_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise EnvironmentError("GROQ_API_KEY is not set.")
    return key


def _groq_headers() -> dict:
    return {"Authorization": f"Bearer {_get_groq_key()}", "Content-Type": "application/json"}


# ─── Key-point extraction ─────────────────────────────────────────────────────

def extract_key_points(text: str) -> list[str]:
    """Return up to 5 short key-point strings extracted from document text."""
    try:
        resp = requests.post(
            GROQ_URL,
            headers=_groq_headers(),
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a document analyst. "
                            "Extract exactly 5 key points from the given text. "
                            'Return ONLY a valid JSON array of 5 short strings, max 8 words each. '
                            'Example: ["Key point one", "Another key point", ...]'
                        ),
                    },
                    {"role": "user", "content": text[:4000]},
                ],
                "max_tokens": 300,
                "temperature": 0.2,
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        match = re.search(r"\[.*?\]", content, re.DOTALL)
        if match:
            points = _json.loads(match.group())
            return [str(p).strip() for p in points if str(p).strip()][:5]
    except Exception:
        pass
    return []


def extract_doc_keywords(text: str) -> list[str]:
    """Return top keywords from document for autocomplete suggestions."""
    try:
        resp = requests.post(
            GROQ_URL,
            headers=_groq_headers(),
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Extract the 15 most important keyword phrases (2-4 words each) "
                            "from this document for a search autocomplete. "
                            "Return ONLY a JSON array of strings."
                        ),
                    },
                    {"role": "user", "content": text[:4000]},
                ],
                "max_tokens": 200,
                "temperature": 0.1,
            },
            timeout=20,
        )
        content = resp.json()["choices"][0]["message"]["content"]
        match = re.search(r"\[.*?\]", content, re.DOTALL)
        if match:
            return [str(k).strip() for k in _json.loads(match.group())][:15]
    except Exception:
        pass
    return []


# ─── Streaming helpers ────────────────────────────────────────────────────────

def _stream_groq(messages: list[dict]):
    """Generator that yields text chunks from a streaming Groq request."""
    try:
        resp = requests.post(
            GROQ_URL,
            headers=_groq_headers(),
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "max_tokens": 2048,
                "temperature": 0.7,
                "stream": True,
            },
            stream=True,
            timeout=90,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8", errors="replace")
            if not decoded.startswith("data: "):
                continue
            payload = decoded[6:]
            if payload.strip() == "[DONE]":
                break
            try:
                chunk = _json.loads(payload)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
            except Exception:
                pass
    except Exception as e:
        yield f"\n\n⚠️ Streaming error: {e}"


def _build_messages(system: str, query: str, history: list[dict] | None) -> list[dict]:
    msgs = [{"role": "system", "content": system}]
    if history:
        for turn in history[-10:]:
            role = turn.get("role", "")
            if role in ("user", "assistant"):
                msgs.append({"role": role, "content": turn["content"]})
    msgs.append({"role": "user", "content": query})
    return msgs


# ─── Highlight helper ─────────────────────────────────────────────────────────

def highlight_text(source_text: str, query: str) -> str:
    """Highlight query keywords in a source text snippet (returns HTML)."""
    words = [w for w in re.split(r"\s+", query) if len(w) > 3]
    result = source_text
    for word in words:
        result = re.sub(
            rf"\b{re.escape(word)}\b",
            (
                f'<mark style="background:rgba(99,102,241,0.25);color:#c4b5fd;'
                f'border-radius:3px;padding:1px 4px;font-weight:700">{word}</mark>'
            ),
            result,
            flags=re.IGNORECASE,
        )
    return result


# ─── RAG Prompts ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT_GENERAL = (
    "You are NexusAI, a highly knowledgeable AI assistant. "
    "Provide a simple, short, direct, and straight-to-the-point answer. "
    "Limit your response to 2 or 3 sentences maximum. "
    "Do NOT use **bold** highlights on more than 2 key words or phrases."
)

SYSTEM_PROMPT_GENERAL_GENZ = (
    "You are NexusAI, a knowledgeable AI assistant speaking in Gen Z/Alpha brain rot slang. "
    "Use slang terms like: rizz, aura, cooked, fr fr, no cap, sigma, skibidi, gyatt, chat, sus, slays, bussin. "
    "Keep your answer simple, short, and straight-to-the-point (2-3 sentences max). "
    "Do NOT use **bold** on more than 2 key words or phrases."
)

SYSTEM_PROMPT_RAG = (
    "You are NexusAI, a precise document-analysis AI. "
    "Answer the user's question using ONLY the provided document context. "
    "Keep your response simple, short, direct, and straight-to-the-point (2-3 sentences max). "
    "Do NOT use **bold** on more than 2 key words or phrases. "
    "If the context doesn't contain the answer, say so clearly. "
    "Cite which source(s) you used at the end of your answer."
)

SYSTEM_PROMPT_RAG_GENZ = (
    "You are NexusAI, a precise document-analysis AI speaking in Gen Z/Alpha brain rot slang. "
    "Answer the user's question using ONLY the provided document context, but translate the explanation "
    "into short Gen Z slang (rizz, aura, cooked, fr fr, no cap, sigma, chat, skibidi). "
    "Keep the answer simple, short, and straight-to-the-point (2-3 sentences max). "
    "Do NOT use **bold** on more than 2 key words or phrases. "
    "If the context doesn't contain the answer, tell chat that it's cooked or not in the docs. "
    "Cite which source(s) you used at the end."
)


class RAGPipeline:
    def __init__(self, index_dir: str = "./index", model_name: str = "all-MiniLM-L6-v2"):
        import os as _os
        index_path = _os.path.join(index_dir, "index.faiss")
        chunks_path = _os.path.join(index_dir, "chunks.pkl")

        if not _os.path.exists(index_path):
            raise FileNotFoundError(
                f"No index found at {index_path}. Run `python ingest.py` first."
            )

        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            data = pickle.load(f)
        self.chunks   = data["chunks"]
        self.metadata = data["metadata"]
        self.embedder = SentenceTransformer(model_name)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self, query: str, top_k: int = 4,
        index=None, chunks=None, metadata=None,
    ) -> list[dict]:
        index    = self.index    if index    is None else index
        chunks   = self.chunks   if chunks   is None else chunks
        metadata = self.metadata if metadata is None else metadata

        import numpy as _np
        query_vec = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)
        scores, indices = index.search(query_vec, top_k)

        return [
            {
                "text":   chunks[idx],
                "source": metadata[idx]["source"],
                "score":  float(score),
            }
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]

    # ── Streaming generators ──────────────────────────────────────────────────

    def stream_general_answer(self, query: str, history: list[dict] | None = None, ai_mode: str = "professional"):
        """Yield text chunks for a general (no-doc) answer."""
        system = SYSTEM_PROMPT_GENERAL_GENZ if ai_mode == "genz" else SYSTEM_PROMPT_GENERAL
        yield from _stream_groq(_build_messages(system, query, history))

    def stream_rag_answer(
        self, query: str, retrieved: list[dict], history: list[dict] | None = None, ai_mode: str = "professional"
    ):
        """Yield text chunks for a document-grounded answer."""
        context = "\n\n---\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in retrieved
        )
        user_content = f"Document context:\n{context}\n\nQuestion: {query}"
        system = SYSTEM_PROMPT_RAG_GENZ if ai_mode == "genz" else SYSTEM_PROMPT_RAG
        yield from _stream_groq(_build_messages(system, user_content, history))

    # ── Non-streaming ─────────────────────────────────────────────────────────

    def general_answer(self, query: str, history: list[dict] | None = None, ai_mode: str = "professional") -> str:
        return "".join(self.stream_general_answer(query, history, ai_mode))

    def generate(
        self, query: str, retrieved_chunks: list[dict], history: list[dict] | None = None, ai_mode: str = "professional"
    ) -> str:
        return "".join(self.stream_rag_answer(query, retrieved_chunks, history, ai_mode))

    def query(
        self, question: str, top_k: int = 4,
        index=None, chunks=None, metadata=None,
        history: list[dict] | None = None,
        ai_mode: str = "professional",
    ) -> dict:
        retrieved = self.retrieve(question, top_k=top_k, index=index, chunks=chunks, metadata=metadata)
        answer    = self.generate(question, retrieved, history, ai_mode)
        return {"answer": answer, "sources": retrieved}