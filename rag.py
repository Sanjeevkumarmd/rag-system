"""
rag.py — Core retrieval-augmented generation logic.

Retrieves the top-k most relevant chunks for a query, then passes them
as context to an LLM to generate a grounded answer.
"""

import os
import pickle

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# Which free LLM backend to use: "ollama" (fully local, no key) or "groq" (free API key).
# Set via environment variable, e.g.: export LLM_BACKEND=ollama
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")


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

    def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        query_vec = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.chunks[idx],
                "source": self.metadata[idx]["source"],
                "score": float(score),
            })
        return results

    def generate(self, query: str, retrieved_chunks: list[dict]) -> str:
        context = "\n\n---\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks
        )

        system_prompt = (
            "You are a helpful assistant. Answer the user's question using ONLY the "
            "provided context. If the context doesn't contain the answer, say so clearly "
            "instead of guessing. Cite which source(s) you used."
        )
        user_content = f"Context:\n{context}\n\nQuestion: {query}"

        if LLM_BACKEND == "ollama":
            return self._generate_ollama(system_prompt, user_content)
        elif LLM_BACKEND == "groq":
            return self._generate_groq(system_prompt, user_content)
        else:
            raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}. Use 'ollama' or 'groq'.")

    def _generate_ollama(self, system_prompt: str, user_content: str) -> str:
        """
        Fully local, fully free. Requires Ollama installed and running
        (https://ollama.com), and a model pulled, e.g.:
            ollama pull llama3.2:3b
        """
        model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def _generate_groq(self, system_prompt: str, user_content: str) -> str:
        """
        Free hosted API. Get a free key (no credit card) at https://console.groq.com
        then: export GROQ_API_KEY=your_key
        """
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("Set GROQ_API_KEY to use the groq backend.")

        model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def query(self, question: str, top_k: int = 4) -> dict:
        retrieved = self.retrieve(question, top_k=top_k)
        answer = self.generate(question, retrieved)
        return {"answer": answer, "sources": retrieved}


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
