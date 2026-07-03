"""
app.py — Friendly, animated Streamlit UI for the RAG system.
Supports per-user document uploads answered in isolation (not shared/saved).

Run with:
    streamlit run app.py
"""

import os
import time
import streamlit as st
import streamlit.components.v1 as components

# --- Cloud deployment shim ---
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        os.environ["LLM_BACKEND"] = "groq"
except FileNotFoundError:
    pass  # No secrets.toml — running locally, skip

from rag import RAGPipeline
import ingest

st.set_page_config(page_title="Ask My Documents", page_icon="🧠", layout="centered")

if not os.path.exists("./index/index.faiss"):
    with st.spinner("First-time setup: building document index..."):
        ingest.build_index("./data", "./index")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0a0e0d; --bg-panel: #101512; --bg-panel-hover: #131a16;
    --line: #1f2b25; --accent: #3ddc84; --accent2: #92fe9d; --amber: #e8a854;
    --text: #e6ebe8; --text-dim: #8a9690; --sans: 'Inter', sans-serif;
}
.stApp {
    background: radial-gradient(circle at 15% 0%, rgba(61,220,132,0.08), transparent 45%), var(--bg);
    font-family: var(--sans);
}
#MainMenu, footer, header { visibility: hidden; }

.hero-wrap { text-align: center; padding: 10px 0 24px 0; animation: fadeInUp 0.6s ease both; }
.hero-title {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 10px 0 6px 0;
}
.hero-sub { color: var(--text-dim); font-size: 1.05rem; margin: 0; }

.stTextInput input {
    background-color: var(--bg-panel) !important; border: 1px solid var(--line) !important;
    border-radius: 10px !important; padding: 14px 16px !important;
    font-size: 15px !important; color: var(--text) !important;
}
.stTextInput input:focus {
    border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(61,220,132,0.12) !important;
}

.stButton button {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    color: #06120b !important; border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; padding: 10px 0 !important; transition: transform 0.15s ease;
}
.stButton button:hover { transform: translateY(-1px); box-shadow: 0 0 16px rgba(61,220,132,0.3); }

.example-btn button {
    background: var(--bg-panel) !important; color: var(--text) !important;
    border: 1px solid var(--line) !important; font-weight: 500 !important; font-size: 13px !important;
}
.example-btn button:hover { border-color: var(--accent) !important; }

.upload-panel {
    background: var(--bg-panel); border: 1px dashed var(--line);
    border-radius: 12px; padding: 18px 20px; margin-bottom: 18px;
}
.upload-active {
    background: rgba(61,220,132,0.07); border: 1px solid var(--accent);
    border-radius: 10px; padding: 12px 16px; margin-bottom: 16px;
    font-size: 13.5px; color: var(--accent); display: flex;
    justify-content: space-between; align-items: center;
}

.confidence-pill {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 12px; padding: 4px 12px; border-radius: 20px; margin-left: 10px;
}
.confidence-high { background: rgba(61,220,132,0.12); color: var(--accent); border: 1px solid rgba(61,220,132,0.3); }
.confidence-med  { background: rgba(232,168,84,0.12); color: var(--amber); border: 1px solid rgba(232,168,84,0.3); }
.confidence-low  { background: rgba(220,80,80,0.12); color: #e07070; border: 1px solid rgba(220,80,80,0.3); }
.confidence-dot { width: 5px; height: 5px; border-radius: 50%; background: currentColor; }

.answer-label {
    font-size: 12px; color: var(--accent); letter-spacing: 1px;
    text-transform: uppercase; font-weight: 700;
}

.sources-label {
    font-size: 12px; color: var(--text-dim); letter-spacing: 1px;
    text-transform: uppercase; margin: 22px 0 10px 0; font-weight: 600;
}
.source-card {
    background: var(--bg-panel); border: 1px solid var(--line);
    border-radius: 8px; padding: 14px 16px; margin-bottom: 10px;
    transition: border-color 0.2s ease;
}
.source-card:hover { border-color: var(--accent); }
.source-top { display: flex; justify-content: space-between; align-items: center; }
.source-title { color: var(--text); font-weight: 600; font-size: 13px; }
.score-badge {
    background: rgba(61,220,132,0.1); color: var(--accent);
    border: 1px solid rgba(61,220,132,0.25); border-radius: 20px;
    padding: 3px 10px; font-size: 11px;
}
.source-text { margin-top: 8px; color: var(--text-dim); font-size: 13.5px; line-height: 1.6; }

.footer-note {
    text-align: center; color: var(--text-dim); font-size: 13px;
    margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--line);
}
.footer-note a { color: var(--accent); text-decoration: none; }

@keyframes fadeInUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
</style>
""", unsafe_allow_html=True)

# ---- Hero ----
st.markdown("""
<div class="hero-wrap">
    <div style="font-size:52px;">🧠</div>
    <p class="hero-title">Ask My Documents</p>
    <p class="hero-sub">Upload your own file, or ask about the built-in knowledge base — either way, answers are grounded in real text.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    return RAGPipeline()

try:
    with st.spinner("Loading models..."):
        pipeline = load_pipeline()
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Run `python ingest.py` first to build the index.")
    st.stop()

# ---- Session state for per-user uploaded docs ----
if "user_index" not in st.session_state:
    st.session_state.user_index = None
    st.session_state.user_chunks = None
    st.session_state.user_metadata = None
    st.session_state.user_filenames = []

# ---- Main-page upload box: private to this browser session only ----
st.markdown('<div class="upload-panel">', unsafe_allow_html=True)
st.markdown("**📤 Upload your own document(s)** — answers will come only from what you upload here.")
uploads = st.file_uploader(
    "Drop .txt or .pdf files",
    type=["txt", "pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)
build_col, clear_col = st.columns([1, 1])
with build_col:
    build_clicked = st.button("📚 Use these documents", use_container_width=True)
with clear_col:
    clear_clicked = st.button("🗑️ Clear & use default knowledge base", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if build_clicked and uploads:
    docs = []
    for f in uploads:
        text = ingest.extract_text_from_bytes(f.getvalue(), f.name)
        if text.strip():
            docs.append({"text": text, "source": f.name})
    if not docs:
        st.error("Couldn't extract any text from those files.")
    else:
        with st.spinner("Reading and indexing your document(s)..."):
            idx, chunks, meta = ingest.build_index_from_docs(docs, pipeline.embedder)
        if idx is None:
            st.error("No usable text found in the uploaded file(s).")
        else:
            st.session_state.user_index = idx
            st.session_state.user_chunks = chunks
            st.session_state.user_metadata = meta
            st.session_state.user_filenames = [f.name for f in uploads]
            st.success(f"Ready! Answering from {len(uploads)} uploaded file(s) now.")

if clear_clicked:
    st.session_state.user_index = None
    st.session_state.user_chunks = None
    st.session_state.user_metadata = None
    st.session_state.user_filenames = []
    st.success("Switched back to the default knowledge base.")

using_user_docs = st.session_state.user_index is not None

if using_user_docs:
    st.markdown(f"""
    <div class="upload-active">
        <span>📎 Answering from your upload: {', '.join(st.session_state.user_filenames)}</span>
    </div>
    """, unsafe_allow_html=True)

# ---- Example question buttons ----
if "prefill" not in st.session_state:
    st.session_state.prefill = ""

st.markdown("**💡 Try one of these:**")
ex_col1, ex_col2, ex_col3 = st.columns(3)
with ex_col1:
    st.markdown('<div class="example-btn">', unsafe_allow_html=True)
    if st.button("What is RAG?", use_container_width=True):
        st.session_state.prefill = "What is RAG?"
    st.markdown('</div>', unsafe_allow_html=True)
with ex_col2:
    st.markdown('<div class="example-btn">', unsafe_allow_html=True)
    if st.button("Summarize this", use_container_width=True):
        st.session_state.prefill = "Summarize the main points"
    st.markdown('</div>', unsafe_allow_html=True)
with ex_col3:
    st.markdown('<div class="example-btn">', unsafe_allow_html=True)
    if st.button("Key takeaways?", use_container_width=True):
        st.session_state.prefill = "What are the key takeaways?"
    st.markdown('</div>', unsafe_allow_html=True)

question = st.text_input(
    "", value=st.session_state.prefill,
    placeholder="Ask something about your documents...",
    label_visibility="collapsed"
)
col1, col2 = st.columns([1, 5])
with col1:
    ask = st.button("Ask →", use_container_width=True)

if question and (ask or question):
    start = time.time()
    with st.spinner("Reading documents and writing your answer..."):
        if using_user_docs:
            result = pipeline.query(
                question,
                index=st.session_state.user_index,
                chunks=st.session_state.user_chunks,
                metadata=st.session_state.user_metadata,
            )
        else:
            result = pipeline.query(question)
    elapsed = time.time() - start

    top_score = result["sources"][0]["score"] if result["sources"] else 0
    if top_score >= 0.6:
        conf_class, conf_label = "confidence-high", "STRONG MATCH"
    elif top_score >= 0.4:
        conf_class, conf_label = "confidence-med", "PARTIAL MATCH"
    else:
        conf_class, conf_label = "confidence-low", "WEAK MATCH"

    model_name = os.environ.get(
        "OLLAMA_MODEL" if os.environ.get("LLM_BACKEND", "ollama") == "ollama" else "GROQ_MODEL",
        "llama3.2:3b" if os.environ.get("LLM_BACKEND", "ollama") == "ollama" else "llama-3.1-8b-instant",
    )

    st.markdown(f"""
    <div style="margin-top:20px; display:flex; align-items:center;">
        <span class="answer-label">✅ Answer</span>
        <span class="confidence-pill {conf_class}"><span class="confidence-dot"></span>{conf_label}</span>
    </div>
    """, unsafe_allow_html=True)

    safe_answer = (
        result["answer"]
        .replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("</script", "<\\/script")
    )
    tw_height = 60 + (len(result["answer"]) // 90) * 26
    components.html(f"""
    <div id="tw-box" style="
        font-family: 'Inter', sans-serif; font-size: 16px; line-height: 1.75;
        color: #e6ebe8; background: #101512; border: 1px solid #1f2b25;
        border-left: 3px solid #3ddc84; border-radius: 10px;
        padding: 20px 24px; min-height: {tw_height}px;
    "></div>
    <script>
        const text = `{safe_answer}`;
        const box = document.getElementById('tw-box');
        let i = 0;
        function type() {{
            if (i <= text.length) {{
                box.innerHTML = text.slice(0, i).replace(/\\n/g, '<br>');
                i += Math.max(1, Math.floor(text.length / 200));
                setTimeout(type, 12);
            }}
        }}
        type();
    </script>
    """, height=tw_height + 20)

    st.markdown('<div class="sources-label">📄 Where this came from</div>', unsafe_allow_html=True)
    for i, s in enumerate(result["sources"], start=1):
        st.markdown(f"""
        <div class="source-card">
            <div class="source-top">
                <span class="source-title">📎 {s['source']}</span>
                <span class="score-badge">match {s['score']:.0%}</span>
            </div>
            <div class="source-text">{s['text'][:280]}...</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔧 Technical details"):
        st.write(f"**Response time:** {elapsed:.2f}s")
        st.write(f"**Chunks retrieved:** {len(result['sources'])}")
        st.write(f"**Top similarity score:** {top_score:.3f}")
        st.write(f"**Model:** {model_name}")

st.markdown("""
<div class="footer-note">
    Built by Sanjeev Kumar M D · <a href="https://github.com/Sanjeevkumarmd" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)