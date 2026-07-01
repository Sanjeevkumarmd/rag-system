"""
app.py — Animated, terminal-inspired Streamlit UI for the RAG system.

Run with:
    streamlit run app.py
"""

import os
import time
import streamlit as st
import streamlit.components.v1 as components

# --- Cloud deployment shim ---
# On Streamlit Community Cloud, secrets are set via the dashboard, not a .env file.
# This pulls them into os.environ so rag.py (which reads os.environ) picks them up
# automatically, with no changes needed to rag.py itself.
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    os.environ["LLM_BACKEND"] = "groq"

from rag import RAGPipeline
import ingest

st.set_page_config(page_title="RAG Assistant", page_icon="◉", layout="centered")

# Build the index automatically on first run if it doesn't exist yet
# (Streamlit Cloud starts from a fresh checkout each time, so ./index won't be there
# unless it's committed to the repo).
if not os.path.exists("./index/index.faiss"):
    with st.spinner("First-time setup: building document index..."):
        ingest.build_index("./data", "./index")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #0a0e0d;
    --bg-panel: #101512;
    --bg-panel-hover: #131a16;
    --line: #1f2b25;
    --accent: #3ddc84;
    --accent-dim: #2a9c5f;
    --amber: #e8a854;
    --text: #e6ebe8;
    --text-dim: #8a9690;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(61,220,132,0.06), transparent 45%),
        var(--bg);
}

#MainMenu, footer, header { visibility: hidden; }

/* ---- Title block with scanning line signature ---- */
.hero {
    padding: 8px 0 20px 0;
    animation: fadeInUp 0.6s ease both;
}
.hero .eyebrow {
    font-family: var(--mono);
    font-size: 12px;
    letter-spacing: 3px;
    color: var(--accent);
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 8px;
}
.hero .eyebrow .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
    animation: blink 1.8s ease-in-out infinite;
}
.hero h1 {
    font-family: var(--sans);
    font-weight: 600;
    font-size: 34px;
    color: var(--text);
    margin: 6px 0 4px 0;
    letter-spacing: -0.5px;
}
.hero p {
    font-family: var(--sans);
    color: var(--text-dim);
    font-size: 14.5px;
    margin: 0;
}
.scan-line {
    margin-top: 18px;
    height: 1px;
    background: var(--line);
    position: relative;
    overflow: hidden;
}
.scan-line::after {
    content: "";
    position: absolute;
    top: 0; left: -30%;
    width: 30%; height: 100%;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    animation: sweep 3.2s ease-in-out infinite;
}

/* ---- Input ---- */
.stTextInput input {
    background-color: var(--bg-panel) !important;
    border: 1px solid var(--line) !important;
    border-radius: 6px !important;
    padding: 14px 16px !important;
    font-family: var(--sans) !important;
    font-size: 15px !important;
    color: var(--text) !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(61,220,132,0.12) !important;
}
.stTextInput input::placeholder { color: var(--text-dim) !important; }

.stButton button {
    background-color: var(--accent) !important;
    color: #06120b !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: var(--mono) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.5px;
    padding: 10px 0 !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton button:hover {
    box-shadow: 0 0 16px rgba(61,220,132,0.35);
    transform: translateY(-1px);
}

/* ---- Stats bar ---- */
.stats-bar {
    display: flex;
    gap: 0;
    background: var(--bg-panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    margin-top: 16px;
    overflow: hidden;
    animation: fadeInUp 0.4s ease both;
}
.stat {
    flex: 1;
    padding: 12px 16px;
    border-right: 1px solid var(--line);
}
.stat:last-child { border-right: none; }
.stat-label {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-dim);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.stat-value {
    font-family: var(--mono);
    font-size: 16px;
    font-weight: 600;
    color: var(--text);
}
.stat-value.accent { color: var(--accent); }
.stat-value.amber { color: var(--amber); }

/* ---- Confidence pill ---- */
.confidence-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: var(--mono);
    font-size: 11px;
    padding: 4px 12px;
    border-radius: 20px;
    margin-left: 10px;
}
.confidence-high { background: rgba(61,220,132,0.12); color: var(--accent); border: 1px solid rgba(61,220,132,0.3); }
.confidence-med  { background: rgba(232,168,84,0.12); color: var(--amber); border: 1px solid rgba(232,168,84,0.3); }
.confidence-low  { background: rgba(220,80,80,0.12); color: #e07070; border: 1px solid rgba(220,80,80,0.3); }
.confidence-dot { width: 5px; height: 5px; border-radius: 50%; background: currentColor; }

/* ---- Typewriter answer ---- */
.tw-cursor {
    display: inline-block;
    width: 7px;
    height: 15px;
    background: var(--accent);
    margin-left: 2px;
    animation: blink 0.9s step-end infinite;
    vertical-align: text-bottom;
}
.answer-box {
    background: var(--bg-panel);
    border: 1px solid var(--line);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 22px 24px;
    margin-top: 18px;
    font-family: var(--sans);
    font-size: 15.5px;
    line-height: 1.7;
    color: var(--text);
    animation: fadeInUp 0.5s ease both;
}
.answer-label {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--accent);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
    display: block;
}

/* ---- Sources ---- */
.sources-label {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-dim);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 26px 0 12px 0;
}
.source-card {
    background: var(--bg-panel);
    border: 1px solid var(--line);
    border-radius: 7px;
    padding: 14px 16px;
    margin-bottom: 10px;
    transition: background 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
    animation: fadeInUp 0.45s ease both;
}
.source-card:hover {
    background: var(--bg-panel-hover);
    border-color: var(--accent-dim);
    transform: translateX(2px);
}
.source-top { display: flex; justify-content: space-between; align-items: center; }
.source-rank {
    font-family: var(--mono);
    color: var(--text-dim);
    font-size: 11px;
    margin-right: 8px;
}
.source-title {
    font-family: var(--mono);
    color: var(--text);
    font-weight: 500;
    font-size: 13px;
}
.score-badge {
    font-family: var(--mono);
    background: rgba(61,220,132,0.1);
    color: var(--accent);
    border: 1px solid rgba(61,220,132,0.25);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
}
.source-text {
    margin-top: 9px;
    color: var(--text-dim);
    font-family: var(--sans);
    font-size: 13.5px;
    line-height: 1.6;
}

.hint {
    font-family: var(--mono);
    color: var(--text-dim);
    font-size: 13px;
    border: 1px dashed var(--line);
    border-radius: 6px;
    padding: 14px 16px;
    margin-top: 20px;
}
.hint b { color: var(--amber); }

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.25; }
}
@keyframes sweep {
    0%   { left: -30%; }
    100% { left: 130%; }
}
</style>
""", unsafe_allow_html=True)

# ---- Hero ----
st.markdown("""
<div class="hero">
    <div class="eyebrow"><span class="dot"></span>LOCAL · PRIVATE · OFFLINE-CAPABLE</div>
    <h1>RAG Assistant</h1>
    <p>Retrieval-augmented answers, grounded in your own documents.</p>
    <div class="scan-line"></div>
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

question = st.text_input(
    "", placeholder="Ask something about your documents...",
    label_visibility="collapsed"
)
col1, col2 = st.columns([1, 5])
with col1:
    ask = st.button("ASK →", use_container_width=True)

if question and (ask or question):
    start = time.time()
    with st.spinner("Scanning documents and generating answer..."):
        result = pipeline.query(question)
    elapsed = time.time() - start

    top_score = result["sources"][0]["score"] if result["sources"] else 0
    if top_score >= 0.6:
        conf_class, conf_label = "confidence-high", "HIGH MATCH"
    elif top_score >= 0.4:
        conf_class, conf_label = "confidence-med", "MODERATE MATCH"
    else:
        conf_class, conf_label = "confidence-low", "LOW MATCH"

    model_name = os.environ.get(
        "OLLAMA_MODEL" if os.environ.get("LLM_BACKEND", "ollama") == "ollama" else "GROQ_MODEL",
        "llama3.2:3b" if os.environ.get("LLM_BACKEND", "ollama") == "ollama" else "llama-3.1-8b-instant",
    )

    st.markdown(f"""
    <div class="stats-bar">
        <div class="stat">
            <div class="stat-label">Latency</div>
            <div class="stat-value accent">{elapsed:.2f}s</div>
        </div>
        <div class="stat">
            <div class="stat-label">Chunks retrieved</div>
            <div class="stat-value">{len(result["sources"])}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Top similarity</div>
            <div class="stat-value amber">{top_score:.3f}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Model</div>
            <div class="stat-value" style="font-size:13px">{model_name}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:16px; display:flex; align-items:center;">
        <span class="answer-label" style="margin-bottom:0">◉ Answer</span>
        <span class="confidence-pill {conf_class}"><span class="confidence-dot"></span>{conf_label}</span>
    </div>
    """, unsafe_allow_html=True)

    # Typewriter reveal — real JS animation, runs inside an iframe component
    safe_answer = (
        result["answer"]
        .replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("</script", "<\\/script")
    )
    tw_height = 60 + (len(result["answer"]) // 90) * 26
    components.html(f"""
    <div id="tw-box" style="
        font-family: 'Inter', sans-serif;
        font-size: 15.5px;
        line-height: 1.7;
        color: #e6ebe8;
        background: #101512;
        border: 1px solid #1f2b25;
        border-left: 3px solid #3ddc84;
        border-radius: 8px;
        padding: 20px 24px;
        min-height: {tw_height}px;
    "></div>
    <span id="tw-cursor" style="
        display:inline-block; width:7px; height:15px;
        background:#3ddc84; animation: blink 0.9s step-end infinite;
    "></span>
    <style>
        @keyframes blink {{ 0%,100% {{opacity:1;}} 50% {{opacity:0.2;}} }}
    </style>
    <script>
        const text = `{safe_answer}`;
        const box = document.getElementById('tw-box');
        let i = 0;
        function type() {{
            if (i <= text.length) {{
                box.innerHTML = text.slice(0, i).replace(/\\n/g, '<br>');
                i += Math.max(1, Math.floor(text.length / 200));
                setTimeout(type, 12);
            }} else {{
                document.getElementById('tw-cursor').style.display = 'none';
            }}
        }}
        type();
    </script>
    """, height=tw_height + 20)

    st.markdown('<div class="sources-label">▸ Retrieved sources</div>', unsafe_allow_html=True)

    for i, s in enumerate(result["sources"], start=1):
        st.markdown(f"""
        <div class="source-card" style="animation-delay:{i * 0.08}s">
            <div class="source-top">
                <span><span class="source-rank">#{i:02d}</span><span class="source-title">{s['source']}</span></span>
                <span class="score-badge">sim {s['score']:.3f}</span>
            </div>
            <div class="source-text">{s['text'][:280]}...</div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="hint">Try asking: <b>What is RAG?</b></div>
    """, unsafe_allow_html=True)
