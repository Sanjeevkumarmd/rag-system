"""
app.py — Streamlit UI for the RAG system.

Run with:
    streamlit run app.py
"""

import streamlit as st
from rag import RAGPipeline

st.set_page_config(page_title="RAG Demo", page_icon="🔍")
st.title("🔍 RAG System Demo")
st.caption("Retrieval-Augmented Generation over your own documents")

@st.cache_resource
def load_pipeline():
    return RAGPipeline()

try:
    pipeline = load_pipeline()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

question = st.text_input("Ask a question about your documents:")

if question:
    with st.spinner("Retrieving relevant chunks and generating answer..."):
        result = pipeline.query(question)

    st.subheader("Answer")
    st.write(result["answer"])

    with st.expander("Retrieved sources (what the model actually saw)"):
        for s in result["sources"]:
            st.markdown(f"**{s['source']}** — similarity: `{s['score']:.3f}`")
            st.text(s["text"][:400] + "...")
            st.divider()
