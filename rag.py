"""
rag.py — Core retrieval-augmented generation logic.

Retrieves the top-k most relevant chunks for a query, then passes them
as context to an LLM to generate a grounded answer.
Also provides a general_answer() for open-ended questions with no docs.
"""

import os
import pickle

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

LLM_BACKEND = os.environ.get("LLM_BACKEND", "groq")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def _get_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to .streamlit/secrets.toml or as an env var."
        )
    return key


class RAGPipeline:
    def __init__(self, index_dir: str = "./index", model_name: str = "all-MiniLM-L6-v2"):
        index_path = os.path.join(index_dir, "index.faiss")
        chunks_path = os.path.join(index_dir, "chunks.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"No index found at {index_path}. Run `python ingest.py` first."
            )

        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.metadata = data["metadata"]

        self.embedder = SentenceTransformer(model_name)

    def retrieve(self, query: str, top_k: int = 4, index=None, chunks=None, metadata=None) -> list[dict]:
        """Retrieve top_k chunks from the FAISS index."""
        index = self.index if index is None else index
        chunks = self.chunks if chunks is None else chunks
        metadata = self.metadata if metadata is None else metadata

        query_vec = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)

        scores, indices = index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": chunks[idx],
                "source": metadata[idx]["source"],
                "score": float(score),
            })
        return results

    # ── Document-grounded generation ──────────────────────────────────────────

    def generate(self, query: str, retrieved_chunks: list[dict], history: list[dict] | None = None) -> str:
        context = "\n\n---\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks
        )

        system_prompt = (
            "You are NexusAI, a precise and helpful AI assistant. "
            "Answer the user's question using ONLY the provided document context. "
            "If the context doesn't contain the answer, say so clearly. "
            "Cite which source(s) you used at the end of your answer."
        )
        user_content = f"Document context:\n{context}\n\nQuestion: {query}"
        return self._call_groq(system_prompt, user_content, history)

    # ── General (no-doc) generation ───────────────────────────────────────────

    def general_answer(self, query: str, history: list[dict] | None = None) -> str:
        """Answer any question from training knowledge — no document needed."""
        system_prompt = (
            "You are NexusAI, a brilliant and knowledgeable AI assistant. "
            "You have deep expertise in every field — technology, science, medicine, "
            "business, law, history, arts, and more. "
            "Answer questions thoroughly, clearly, and helpfully. "
            "Use markdown formatting (headers, bullet points, code blocks) when it improves clarity. "
            "Be concise but complete."
        )
        return self._call_groq(system_prompt, query, history)

    # ── Groq API call ─────────────────────────────────────────────────────────

    def _call_groq(self, system_prompt: str, user_content: str, history: list[dict] | None = None) -> str:
        api_key = _get_groq_key()
        model = GROQ_MODEL

        messages = [{"role": "system", "content": system_prompt}]

        # Inject up to 10 previous turns for context
        if history:
            for turn in history[-10:]:
                role = turn.get("role", "user")
                if role in ("user", "assistant"):
                    messages.append({"role": role, "content": turn["content"]})

        messages.append({"role": "user", "content": user_content})

        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 2048,
                    "temperature": 0.7,
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.HTTPError as e:
            return f"⚠️ API error ({resp.status_code}): {resp.text[:300]}"
        except Exception as e:
            return f"⚠️ Error calling Groq: {e}"

    # ── Full query pipeline ────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int = 4,
        index=None,
        chunks=None,
        metadata=None,
        history: list[dict] | None = None,
    ) -> dict:
        retrieved = self.retrieve(question, top_k=top_k, index=index, chunks=chunks, metadata=metadata)
        answer = self.generate(question, retrieved, history)
        return {"answer": answer, "sources": retrieved}


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = RAGPipeline()
    while True:
        q = input("\nAsk a question (or 'quit'): ")
        if q.lower() == "quit":
            break
        result = pipeline.query(q)
        print(f"\nAnswer: {result['answer']}")
        print("\nSources used:")
        for s in result["sources"]:
            print(f"  - {s['source']} (score: {s['score']:.3f})")