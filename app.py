"""
app.py — NexusAI: Premium AI Knowledge Assistant
Powered by Groq (llama-3.3-70b-versatile) + FAISS local RAG.

Features:
  • Chat with any question — general AI knowledge OR uploaded documents
  • 4-hour session windows with 1-hour cooldown (then auto-resume with memory)
  • Conversation history stored per session (survives cooldown breaks)
  • Premium dark glassmorphism UI
  • Deploy to Streamlit Community Cloud for a public URL

Run locally: streamlit run app.py
"""

import os
import time
import math
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# ─────────────────────────────────────────────────────────────────────────────
# Secrets — read GROQ_API_KEY from Streamlit secrets or env
# ─────────────────────────────────────────────────────────────────────────────
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except FileNotFoundError:
    pass  # No secrets.toml — running locally without secrets file

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SESSION_HOURS  = 4
COOLDOWN_HOURS = 1
SESSION_SECS   = SESSION_HOURS  * 3600
COOLDOWN_SECS  = COOLDOWN_HOURS * 3600
GROQ_MODEL     = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be FIRST Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NexusAI — AI Knowledge Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports (after page config)
# ─────────────────────────────────────────────────────────────────────────────
import ingest
from rag import RAGPipeline
from memory import new_session_id, save_session, load_session, session_exists

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS — premium dark glassmorphism
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap');

:root {
  --bg:           #07090f;
  --bg2:          #0d1020;
  --glass:        rgba(255,255,255,0.035);
  --glass-b:      rgba(255,255,255,0.08);
  --indigo:       #6366f1;
  --violet:       #8b5cf6;
  --text:         #e2e8f0;
  --text-dim:     #64748b;
  --green:        #22c55e;
  --amber:        #f59e0b;
  --red:          #f43f5e;
  --sans:         'Inter', system-ui, sans-serif;
}

html, body, * { font-family: var(--sans) !important; box-sizing: border-box; }

/* ── App background ─────────────────────────────── */
.stApp {
  background: radial-gradient(ellipse 80% 60% at 50% -10%,
              rgba(99,102,241,0.15) 0%, transparent 70%),
              radial-gradient(ellipse 50% 40% at 90% 110%,
              rgba(139,92,246,0.1) 0%, transparent 60%),
              var(--bg) !important;
}
.main .block-container {
  padding: 2rem 2.5rem 6rem 2.5rem !important;
  max-width: 900px !important;
  margin: 0 auto !important;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ─────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: rgba(10,12,22,0.97) !important;
  border-right: 1px solid var(--glass-b) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stMarkdown p { color: var(--text-dim) !important; font-size:13px; }
[data-testid="stSidebarNav"] { display: none !important; }

/* ── Sidebar file uploader ───────────────────────── */
[data-testid="stFileUploader"] > div {
  background: var(--glass) !important;
  border: 1px dashed var(--glass-b) !important;
  border-radius: 12px !important;
}

/* ── Chat messages ───────────────────────────────── */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  animation: fadeInUp 0.3s ease both;
}
[data-testid="stChatMessageContent"] {
  background: var(--glass) !important;
  border: 1px solid var(--glass-b) !important;
  border-radius: 16px !important;
  padding: 16px 20px !important;
  color: var(--text) !important;
  font-size: 15px !important;
  line-height: 1.75 !important;
  backdrop-filter: blur(10px) !important;
}

/* ── Chat input box ──────────────────────────────── */
[data-testid="stChatInput"] {
  background: rgba(13,16,32,0.9) !important;
  border-top: 1px solid var(--glass-b) !important;
  padding: 12px 16px !important;
  backdrop-filter: blur(20px) !important;
}
[data-testid="stChatInput"] textarea {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--glass-b) !important;
  border-radius: 14px !important;
  color: var(--text) !important;
  font-size: 15px !important;
  caret-color: var(--indigo) !important;
}
[data-testid="stChatInput"] textarea:focus {
  border-color: var(--indigo) !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,0.18) !important;
  outline: none !important;
}

/* ── Buttons ─────────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg, var(--indigo), var(--violet)) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em !important;
  transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 28px rgba(99,102,241,0.45) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Progress / expander / text inputs ──────────── */
.stProgress > div > div {
  background: linear-gradient(90deg, var(--indigo), var(--violet)) !important;
  border-radius: 999px !important;
}
[data-testid="stExpander"] {
  background: var(--glass) !important;
  border: 1px solid var(--glass-b) !important;
  border-radius: 14px !important;
}
.stTextInput input, .stTextArea textarea {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--glass-b) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
}
.stTextInput input:focus { border-color: var(--indigo) !important; }
.stSelectbox select { background: var(--bg2) !important; color: var(--text) !important; }

/* ── Text & headings ─────────────────────────────── */
h1,h2,h3,h4,h5,h6 { color: var(--text) !important; }
p, span, div, label { color: var(--text) !important; }
code { background: rgba(99,102,241,0.12) !important; color: #a5b4fc !important; border-radius:4px; padding:2px 6px; }
pre code { background: transparent !important; }
pre { background: rgba(0,0,0,0.4) !important; border: 1px solid var(--glass-b) !important; border-radius:12px !important; }

/* ── Spinner ─────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--indigo) !important; }

/* ── Hero & custom components ────────────────────── */
@keyframes fadeInUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
@keyframes pulse    { 0%,100%{opacity:1} 50%{opacity:0.4} }
@keyframes shimmer  { 0%{background-position:200% center} 100%{background-position:-200% center} }

.nexus-hero {
  text-align: center; padding: 20px 0 8px 0;
  animation: fadeInUp 0.6s ease;
}
.nexus-badge {
  display: inline-flex; align-items: center; gap: 8px;
  background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.3);
  border-radius: 99px; padding: 5px 16px; font-size: 12.5px;
  color: #a5b4fc !important; margin-bottom: 20px; letter-spacing: 0.04em;
}
.nexus-badge-dot { width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2s infinite; }
.nexus-title {
  font-size: clamp(2rem, 5vw, 3.2rem); font-weight: 900; line-height: 1.05;
  background: linear-gradient(135deg, #e2e8f0 0%, #a5b4fc 45%, #c4b5fd 80%, #e2e8f0 100%);
  background-size: 200% auto;
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  animation: shimmer 4s linear infinite; margin: 0 0 12px 0;
}
.nexus-sub {
  font-size: 1.05rem; color: #64748b !important; margin: 0 0 28px 0; line-height: 1.6;
}
.nexus-divider {
  height: 1px; background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent);
  margin: 24px 0 28px 0;
}

.session-bar {
  position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
  height: 3px; background: rgba(255,255,255,0.04);
}
.session-bar-fill {
  height: 100%; border-radius: 0 99px 99px 0;
  transition: width 1s linear;
}

.mode-pill {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 12px; padding: 4px 12px; border-radius: 99px;
  font-weight: 600; letter-spacing: 0.03em;
}
.pill-rag      { background:rgba(34,197,94,0.1);  color:#4ade80 !important; border:1px solid rgba(34,197,94,0.25); }
.pill-general  { background:rgba(99,102,241,0.1); color:#a5b4fc !important; border:1px solid rgba(99,102,241,0.3); }
.pill-cooldown { background:rgba(244,63,94,0.1);  color:#fb7185 !important; border:1px solid rgba(244,63,94,0.25);
                 animation: pulse 1.5s infinite; }

.source-chip {
  display: inline-flex; align-items: center; gap: 5px;
  background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.2);
  border-radius: 6px; padding: 4px 10px; font-size: 12px; color: #a5b4fc !important;
  margin: 4px 4px 0 0;
}

.stat-row {
  display: flex; gap: 12px; margin-bottom: 20px;
}
.stat-box {
  flex: 1; background: var(--glass); border: 1px solid var(--glass-b);
  border-radius: 12px; padding: 14px 16px; text-align: center;
}
.stat-val { font-size: 1.4rem; font-weight: 800; color: #a5b4fc !important; }
.stat-lbl { font-size: 11px; color: var(--text-dim) !important; margin-top: 2px; letter-spacing: 0.05em; text-transform: uppercase; }

.cooldown-card {
  background: rgba(244,63,94,0.06); border: 1px solid rgba(244,63,94,0.2);
  border-radius: 20px; padding: 48px 32px; text-align: center;
  animation: fadeInUp 0.5s ease;
}

.tip-box {
  background: rgba(99,102,241,0.06); border: 1px solid rgba(99,102,241,0.18);
  border-radius: 12px; padding: 14px 18px; font-size: 13.5px;
  color: #94a3b8 !important; margin-bottom: 16px; line-height: 1.6;
}
.tip-box strong { color: #a5b4fc !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session-state bootstrap
# ─────────────────────────────────────────────────────────────────────────────
def _init_session():
    params = st.query_params
    sid = params.get("sid", "")

    if "session_id" not in st.session_state:
        if sid and session_exists(sid):
            # Returning user with a valid session
            st.session_state.session_id = sid
            st.session_state.chat_history = load_session(sid)
            st.session_state.returning = True
        else:
            st.session_state.session_id = new_session_id()
            st.session_state.chat_history = []
            st.session_state.returning = False

        st.session_state.session_start  = time.time()
        st.session_state.cooldown_start = None
        st.session_state.user_index     = None
        st.session_state.user_chunks    = None
        st.session_state.user_metadata  = None
        st.session_state.user_filenames = []

    # Keep URL in sync
    st.query_params["sid"] = st.session_state.session_id


_init_session()


# ─────────────────────────────────────────────────────────────────────────────
# Session timing helpers
# ─────────────────────────────────────────────────────────────────────────────
def _get_status():
    """Returns ('active'|'cooldown', remaining_active_secs, remaining_cooldown_secs)."""
    now = time.time()
    if st.session_state.cooldown_start is not None:
        elapsed_cd = now - st.session_state.cooldown_start
        if elapsed_cd >= COOLDOWN_SECS:
            # Cooldown finished — reset
            st.session_state.session_start  = now
            st.session_state.cooldown_start = None
            return "active", SESSION_SECS, 0
        return "cooldown", 0, COOLDOWN_SECS - elapsed_cd
    else:
        elapsed = now - st.session_state.session_start
        if elapsed >= SESSION_SECS:
            st.session_state.cooldown_start = now
            return "cooldown", 0, COOLDOWN_SECS
        return "active", SESSION_SECS - elapsed, 0


def _fmt_time(secs: float) -> str:
    secs = max(0, int(secs))
    h, m = divmod(secs, 3600)
    m, s = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_pipeline():
    if Path("./index/index.faiss").exists():
        return RAGPipeline()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div style="padding:8px 0 20px 0;">
      <div style="font-size:22px;font-weight:900;background:linear-gradient(135deg,#a5b4fc,#c4b5fd);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px;">
        ⚡ NexusAI
      </div>
      <div style="font-size:12px;color:#475569;">AI Knowledge Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    # Session stats
    status, rem_active, rem_cooldown = _get_status()
    pct = rem_active / SESSION_SECS if status == "active" else 0
    bar_color = "#22c55e" if pct > 0.4 else ("#f59e0b" if pct > 0.15 else "#f43f5e")

    st.markdown("""<div style="font-size:11px;letter-spacing:0.08em;color:#475569;
                               text-transform:uppercase;margin-bottom:8px;">SESSION</div>""",
                unsafe_allow_html=True)

    if status == "active":
        st.progress(pct, text=f"⏱ {_fmt_time(rem_active)} remaining")
    else:
        st.markdown(f"""<div class="mode-pill pill-cooldown" style="width:100%;justify-content:center;
                        margin-bottom:8px;">🕐 Cooldown — {_fmt_time(rem_cooldown)}</div>""",
                    unsafe_allow_html=True)

    st.caption(f"Session ID: `{st.session_state.session_id}`")
    st.caption("Bookmark this URL to return to your conversation.")

    st.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin:16px 0'>", unsafe_allow_html=True)

    # API key override
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        st.markdown("**🔑 Groq API Key**")
        api_key = st.text_input("Enter your free key from groq.com",
                                type="password", placeholder="gsk_...",
                                key="user_api_key")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
        st.caption("[Get a free key →](https://console.groq.com)")
        st.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin:16px 0'>", unsafe_allow_html=True)

    # Document upload
    st.markdown("**📄 Upload Documents** *(optional)*")
    st.caption("Upload PDFs or text files to get answers grounded in your content.")
    uploads = st.file_uploader(
        "Drop files here",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    col_use, col_clr = st.columns(2)
    with col_use:
        use_clicked = st.button("📚 Use docs", use_container_width=True)
    with col_clr:
        clr_clicked = st.button("🗑 Clear", use_container_width=True)

    if use_clicked and uploads:
        pipeline_obj = _load_pipeline()
        if pipeline_obj is None:
            st.error("Base pipeline not loaded. Run `python ingest.py` first.")
        else:
            docs = []
            for f in uploads:
                text = ingest.extract_text_from_bytes(f.getvalue(), f.name)
                if text.strip():
                    docs.append({"text": text, "source": f.name})
            if docs:
                with st.spinner("Indexing documents…"):
                    idx, chunks, meta = ingest.build_index_from_docs(docs, pipeline_obj.embedder)
                if idx:
                    st.session_state.user_index    = idx
                    st.session_state.user_chunks   = chunks
                    st.session_state.user_metadata = meta
                    st.session_state.user_filenames = [f.name for f in uploads]
                    st.success(f"✓ {len(uploads)} file(s) indexed")
                else:
                    st.error("No usable text found.")
            else:
                st.error("Couldn't extract text from those files.")

    if clr_clicked:
        st.session_state.user_index    = None
        st.session_state.user_chunks   = None
        st.session_state.user_metadata = None
        st.session_state.user_filenames = []
        st.success("Switched to general AI mode.")

    if st.session_state.user_filenames:
        st.markdown("<div style='margin-top:8px;'>", unsafe_allow_html=True)
        for fn in st.session_state.user_filenames:
            st.markdown(f"""<div class="source-chip">📎 {fn}</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin:16px 0'>", unsafe_allow_html=True)

    # Clear history
    if st.button("🗂 Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        save_session(st.session_state.session_id, [])
        st.rerun()

    st.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin:16px 0'>", unsafe_allow_html=True)

    # About
    st.markdown("""
    <div style="font-size:12px;color:#475569;line-height:1.7;">
      <strong style="color:#64748b">Model:</strong> llama-3.3-70b<br>
      <strong style="color:#64748b">Memory:</strong> Session-based<br>
      <strong style="color:#64748b">Session:</strong> 4 hrs / 1 hr break<br>
      <strong style="color:#64748b">Built by:</strong>
      <a href="https://github.com/Sanjeevkumarmd" target="_blank"
         style="color:#a5b4fc;text-decoration:none;">Sanjeev Kumar M D</a>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Thin progress bar at the very top of the viewport
# ─────────────────────────────────────────────────────────────────────────────
pct_px = int(pct * 100)
bar_color2 = "#22c55e" if pct > 0.4 else ("#f59e0b" if pct > 0.15 else "#f43f5e")
components.html(f"""
<div style="position:fixed;top:0;left:0;right:0;z-index:10000;height:3px;
            background:rgba(255,255,255,0.04);">
  <div style="height:100%;width:{pct_px}%;
              background:linear-gradient(90deg,{bar_color2},{bar_color2}88);
              border-radius:0 99px 99px 0;transition:width 0.5s;"></div>
</div>
""", height=0)


# ─────────────────────────────────────────────────────────────────────────────
# Hero section
# ─────────────────────────────────────────────────────────────────────────────
using_docs = st.session_state.user_index is not None
mode_label = "📄 Document Mode" if using_docs else "🌐 General AI Mode"
mode_class = "pill-rag" if using_docs else "pill-general"

st.markdown(f"""
<div class="nexus-hero">
  <div class="nexus-badge">
    <span class="nexus-badge-dot"></span>
    POWERED BY GROQ · LLAMA 3.3 · 70B
  </div>
  <div class="nexus-title">NexusAI</div>
  <div class="nexus-sub">
    Ask anything — from any file, any topic, any domain.<br>
    Your private AI knowledge assistant.
  </div>
  <span class="mode-pill {mode_class}">{mode_label}</span>
</div>
<div class="nexus-divider"></div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cooldown gate
# ─────────────────────────────────────────────────────────────────────────────
status, rem_active, rem_cooldown = _get_status()

if status == "cooldown":
    h = int(rem_cooldown // 3600)
    m = int((rem_cooldown % 3600) // 60)
    s = int(rem_cooldown % 60)
    st.markdown(f"""
    <div class="cooldown-card">
      <div style="font-size:52px;margin-bottom:16px;">☕</div>
      <div style="font-size:1.8rem;font-weight:800;color:#e2e8f0;margin-bottom:8px;">
        Take a break!
      </div>
      <div style="font-size:1.1rem;color:#64748b;margin-bottom:24px;">
        Your 4-hour session is complete. Rest for 1 hour, then your conversation
        continues right where you left off.
      </div>
      <div style="font-size:2.5rem;font-weight:900;font-variant-numeric:tabular-nums;
                  background:linear-gradient(135deg,#a5b4fc,#c4b5fd);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        {h:02d}:{m:02d}:{s:02d}
      </div>
      <div style="font-size:13px;color:#475569;margin-top:8px;">until your session resumes</div>
    </div>
    """, unsafe_allow_html=True)

    # Auto-refresh every 60 s so the timer updates
    components.html("""<script>setTimeout(()=>window.parent.location.reload(), 60000);</script>""", height=0)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Load pipeline
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading AI models…"):
    pipeline = _load_pipeline()

if pipeline is None:
    st.info("📁 No document index found. Building from sample data…")
    try:
        with st.spinner("Building index…"):
            ingest.build_index("./data", "./index")
        st.rerun()
    except Exception as e:
        st.error(f"Could not build index: {e}")
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Session resume notice
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.get("returning") and st.session_state.chat_history:
    st.markdown("""
    <div class="tip-box">
      <strong>👋 Welcome back!</strong> Your previous conversation has been restored.
      Just keep chatting — your memory continues.
    </div>
    """, unsafe_allow_html=True)
    st.session_state.returning = False

elif not st.session_state.chat_history:
    st.markdown("""
    <div class="tip-box">
      <strong>⚡ Ask anything.</strong> No document needed — NexusAI answers from broad AI knowledge.
      Upload a PDF or text file in the sidebar to ground answers in your content.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Chat history display
# ─────────────────────────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "⚡"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources used", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f'<span class="source-chip">📄 {src["source"]} — '
                        f'<strong>{src["score"]:.0%} match</strong></span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(src["text"][:250] + "…")


# ─────────────────────────────────────────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────────────────────────────────────────
api_key_present = bool(os.environ.get("GROQ_API_KEY") or st.session_state.get("user_api_key"))

placeholder = (
    "Ask anything — tech, science, business, career advice…"
    if not using_docs
    else "Ask a question about your uploaded document(s)…"
)
if not api_key_present:
    placeholder = "⚠️ Add your Groq API key in the sidebar first…"

user_input = st.chat_input(placeholder, disabled=not api_key_present)

if user_input:
    # ── Append user message to history ──
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # ── Generate response ──
    with st.chat_message("assistant", avatar="⚡"):
        history_for_llm = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history[:-1]  # exclude current user msg
        ]

        sources = []
        with st.spinner("Thinking…"):
            if using_docs:
                result = pipeline.query(
                    user_input,
                    index=st.session_state.user_index,
                    chunks=st.session_state.user_chunks,
                    metadata=st.session_state.user_metadata,
                    history=history_for_llm,
                )
                answer  = result["answer"]
                sources = result["sources"]
            else:
                answer = pipeline.general_answer(user_input, history=history_for_llm)

        st.markdown(answer)

        if sources:
            with st.expander("📎 Sources used", expanded=False):
                for src in sources:
                    st.markdown(
                        f'<span class="source-chip">📄 {src["source"]} — '
                        f'<strong>{src["score"]:.0%} match</strong></span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(src["text"][:250] + "…")

    # ── Append assistant message to history ──
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

    # ── Persist to disk ──
    save_session(st.session_state.session_id, st.session_state.chat_history)

    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:48px;padding-top:20px;
            border-top:1px solid rgba(255,255,255,0.06);
            font-size:12.5px;color:#334155;">
  Built by
  <a href="https://github.com/Sanjeevkumarmd" target="_blank"
     style="color:#6366f1;text-decoration:none;font-weight:600;">Sanjeev Kumar M D</a>
  &nbsp;·&nbsp; Powered by
  <a href="https://groq.com" target="_blank" style="color:#6366f1;text-decoration:none;">Groq</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/Sanjeevkumarmd/rag-system" target="_blank"
     style="color:#6366f1;text-decoration:none;">GitHub</a>
</div>
""", unsafe_allow_html=True)