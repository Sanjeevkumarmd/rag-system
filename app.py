"""
app.py — NexusAI v3: Premium AI Knowledge Assistant
New in v3:
  • File upload on main page with upload animation
  • AI-powered key-point suggestion chips (pop up after upload, fade when typing)
  • Real-time Google-style word autocomplete (matches doc keywords as you type)
  • Streaming typing animation for AI answers
  • Bold key terms auto-highlighted in responses
  • Source viewer with keyword highlight — click chip to see exact match
  • 4-hour sessions / 1-hour cooldown / conversation memory
"""

import os, re, time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# ─── Secrets ─────────────────────────────────────────────────────────────────
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except FileNotFoundError:
    pass

# ─── Constants ────────────────────────────────────────────────────────────────
SESSION_HOURS  = 4
COOLDOWN_HOURS = 1
SESSION_SECS   = SESSION_HOURS  * 3600
COOLDOWN_SECS  = COOLDOWN_HOURS * 3600

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAGAI — AI Knowledge Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Lazy imports ─────────────────────────────────────────────────────────────
import ingest
from rag import RAGPipeline, extract_key_points, extract_doc_keywords, highlight_text
from memory import new_session_id, save_session, load_session, session_exists

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
  --bg:        #07090f;
  --bg2:       #0d1020;
  --glass:     rgba(255,255,255,0.03);
  --glass-b:   rgba(255,255,255,0.07);
  --indigo:    #6366f1;
  --violet:    #8b5cf6;
  --green:     #22c55e;
  --amber:     #f59e0b;
  --red:       #f43f5e;
  --text:      #e2e8f0;
  --dim:       #64748b;
  --sans:      -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
html,body,* { font-family:var(--sans)!important; box-sizing:border-box; font-size:13.5px !important; }

/* App background */
.stApp {
  background:
    radial-gradient(ellipse 80% 50% at 50% -5%, rgba(99,102,241,.18) 0%, transparent 65%),
    radial-gradient(ellipse 40% 30% at 95% 105%, rgba(139,92,246,.12) 0%, transparent 55%),
    var(--bg) !important;
}
.main .block-container {
  padding: 0 2rem 5rem 2rem !important;
  max-width: 860px !important;
  margin: 0 auto !important;
}
#MainMenu,footer,header { visibility:hidden; }
[data-testid="stSidebarNav"] { display:none!important; }

/* Sidebar */
[data-testid="stSidebar"] {
  background:rgba(10,12,22,.97)!important;
  border-right:1px solid var(--glass-b)!important;
}
[data-testid="stSidebar"] * { color:var(--text)!important; }
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stMarkdown p { color:var(--dim)!important; font-size:12px!important; }

/* Sidebar Toggle Customization (highly visible, glowing) */
[data-testid="collapsedControl"] {
  background: rgba(13, 16, 32, 0.95) !important;
  border: 1px solid rgba(99, 102, 241, 0.3) !important;
  border-radius: 8px !important;
  padding: 4px !important;
  left: 20px !important;
  top: 20px !important;
  box-shadow: 0 0 15px rgba(99, 102, 241, 0.2) !important;
}
[data-testid="collapsedControl"] button {
  color: var(--text) !important;
}
[data-testid="collapsedControl"] svg {
  fill: var(--text) !important;
}
[data-testid="stSidebar"] button[aria-label="Close sidebar"] {
  background: rgba(255,255,255,0.02) !important;
  border: 1px solid var(--glass-b) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
}

/* Chat bubble aligning & animations */
[data-testid="stChatMessage"] {
  background:transparent!important;
  border:none!important;
  animation:fadeUp .35s ease both;
  display: flex !important;
  width: 100% !important;
  margin-bottom: 12px !important;
}

/* User bubbles (right side) */
div[data-testid="stChatMessage"]:has(.user-marker) {
  flex-direction: row-reverse !important;
}
div[data-testid="stChatMessage"]:has(.user-marker) [data-testid="stChatMessageContent"] {
  background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(139,92,246,0.18)) !important;
  border: 1px solid rgba(139,92,246,0.3) !important;
  border-radius: 16px 16px 4px 16px !important;
  text-align: left !important;
  margin-left: auto !important;
}

/* Assistant bubbles (left side) */
div[data-testid="stChatMessage"]:has(.assistant-marker) [data-testid="stChatMessageContent"] {
  background: var(--glass)!important;
  border: 1px solid var(--glass-b)!important;
  border-radius: 16px 16px 16px 4px !important;
  margin-right: auto !important;
}

[data-testid="stChatMessageContent"] {
  position: relative !important;
  padding: 12px 18px !important;
  color:var(--text)!important;
  line-height:1.65!important;
  backdrop-filter:blur(10px)!important;
  max-width: 80% !important;
  background: var(--glass) !important;
  border: 1px solid var(--glass-b) !important;
  border-radius: 16px !important;
}

/* Actions Menu on Hover */
.chat-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 6px;
  font-size: 11px !important;
  color: var(--dim);
  opacity: 0;
  transition: opacity 0.2s ease;
  justify-content: flex-end;
}
[data-testid="stChatMessageContent"]:hover .chat-actions {
  opacity: 1;
}
.chat-action-btn {
  background: transparent;
  border: none;
  color: var(--indigo);
  cursor: pointer;
  padding: 0;
  font-size: 11px !important;
  font-weight: 600;
}
.chat-action-btn:hover {
  color: var(--violet);
}

/* Bold inside chat = highlighted */
[data-testid="stChatMessageContent"] strong {
  color:#c4b5fd!important;
  font-weight:700!important;
}
[data-testid="stChatMessageContent"] code {
  background:rgba(99,102,241,.15)!important;
  color:#a5b4fc!important;
  border-radius:4px;
  padding:2px 6px;
}

/* Custom keywords for click highlights */
.chat-keyword {
  color: #a5b4fc !important;
  border-bottom: 1.5px dashed rgba(99,102,241,0.4);
  cursor: pointer;
  font-weight: 600;
  transition: all 0.2s;
}
.chat-keyword:hover {
  background: rgba(99,102,241,0.15) !important;
  color: #c4b5fd !important;
}

/* Chat input container styling */
[data-testid="stChatInput"] {
  background: transparent !important;
  border-top: none !important;
  padding: 10px 0 !important;
  max-width: 1100px !important;
  margin: 0 auto !important;
}
[data-testid="stChatInput"] > div {
  max-width: 1100px !important;
  margin: 0 auto !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
  border-radius: 12px !important;
  background: rgba(13, 16, 32, 0.3) !important;
  backdrop-filter: blur(20px) !important;
  transition: border-color 0.25s, box-shadow 0.25s !important;
  animation: borderGlow 4s infinite alternate;
}
[data-testid="stChatInput"] > div:focus-within {
  border-color: var(--indigo) !important;
  box-shadow: 0 0 20px rgba(99, 102, 241, 0.25) !important;
}
[data-testid="stChatInput"] textarea {
  background: transparent !important;
  border: none !important;
  color: var(--text) !important;
  font-size: 13.5px !important;
}

@keyframes borderGlow {
  0% { border-color: rgba(99, 102, 241, 0.15); box-shadow: 0 0 8px rgba(99, 102, 241, 0.03); }
  50% { border-color: rgba(139, 92, 246, 0.4); box-shadow: 0 0 15px rgba(139, 92, 246, 0.12); }
  100% { border-color: rgba(99, 102, 241, 0.15); box-shadow: 0 0 8px rgba(99, 102, 241, 0.03); }
}

/* Redesigned Glassmorphic Buttons */
.stButton>button {
  background: rgba(255, 255, 255, 0.03) !important;
  color: var(--text) !important;
  border: 1px solid var(--glass-b) !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
  box-shadow: none !important;
}
.stButton>button:hover {
  background: rgba(99, 102, 241, 0.1) !important;
  border-color: var(--indigo) !important;
  box-shadow: 0 4px 16px rgba(99, 102, 241, 0.15) !important;
  transform: translateY(-1px) !important;
}
.stButton>button:active {
  transform: translateY(0) scale(0.97) !important;
}

/* Progress */
.stProgress>div>div {
  background:linear-gradient(90deg,var(--indigo),var(--violet))!important;
  border-radius:99px!important;
}

/* Headings / text */
h1,h2,h3,h4 { color:var(--text)!important; }
p,span,div,label { color:var(--text)!important; }
.stMarkdown p { color:var(--text)!important; }
pre { background:rgba(0,0,0,.4)!important; border:1px solid var(--glass-b)!important; border-radius:12px!important; }
[data-testid="stExpander"] {
  background:var(--glass)!important;
  border:1px solid var(--glass-b)!important;
  border-radius:14px!important;
}
[data-testid="stExpander"] summary {
  padding: 10px 14px !important;
}
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] summary span[data-testid="stExpanderIcon"],
[data-testid="stExpander"] summary svg[class*="chevron"] {
  display: none !important;
}
[data-testid="stExpander"] summary::after {
  content: " ▾";
  color: var(--indigo);
  font-size: 14px;
  font-weight: bold;
  margin-left: auto;
  transition: transform 0.2s;
}
[data-testid="stExpander"] summary[aria-expanded="true"]::after {
  content: " ▴";
}

/* File uploader */
[data-testid="stFileUploader"]>div {
  background:var(--glass)!important;
  border:1.5px dashed rgba(99,102,241,.4)!important;
  border-radius:16px!important;
  transition:border-color .2s,background .2s!important;
}
[data-testid="stFileUploader"]>div:hover {
  background:rgba(99,102,241,.06)!important;
  border-color:var(--indigo)!important;
}
[data-testid="stFileUploader"] small {
  display: none !important;
}
[data-testid="stFileUploader"] div[data-testid="stMarkdownContainer"] {
  display: none !important;
}
[data-testid="stFileUploader"] span {
  display: none !important;
}
[data-testid="stFileUploader"] p {
  display: none !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
  display: none !important;
}
[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] > div {
  display: none !important;
}
[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"]::after {
  content: "Drag and drop or click to upload your document";
  color: var(--text);
  font-size: 14px;
}

/* Spinner */
.stSpinner>div { border-top-color:var(--indigo)!important; }

/* ── Custom components ─────────────────────────────────────────── */
@keyframes fadeUp   { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeIn   { from{opacity:0} to{opacity:1} }
@keyframes pulse    { 0%,100%{opacity:1} 50%{opacity:.45} }
@keyframes shimmer  { 0%{background-position:200% center} 100%{background-position:-200% center} }
@keyframes slideIn  { from{opacity:0;transform:translateY(-14px) scale(.95)} to{opacity:1;transform:translateY(0) scale(1)} }
@keyframes popIn    { 0%{opacity:0;transform:scale(.7)} 70%{transform:scale(1.08)} 100%{opacity:1;transform:scale(1)} }
@keyframes checkPop { 0%{transform:scale(0)} 60%{transform:scale(1.3)} 100%{transform:scale(1)} }

.nexus-hero { text-align:center; padding:28px 0 8px 0; animation:fadeUp .6s ease; }
.nexus-badge {
  display:inline-flex; align-items:center; gap:8px;
  background:rgba(99,102,241,.1); border:1px solid rgba(99,102,241,.3);
  border-radius:99px; padding:5px 16px; font-size:12px; color:#a5b4fc!important;
  margin-bottom:20px; letter-spacing:.05em;
}
.nexus-badge-dot { width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2s infinite; }
.nexus-title {
  font-size:clamp(2rem,5vw,3rem); font-weight:900; line-height:1.05;
  background:linear-gradient(135deg,#e2e8f0 0%,#a5b4fc 45%,#c4b5fd 80%,#e2e8f0 100%);
  background-size:200% auto; -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  animation:shimmer 4s linear infinite; margin:0 0 10px 0;
}
.nexus-sub { font-size:1.05rem; color:#64748b!important; margin:0 0 24px 0; line-height:1.6; }
.nexus-divider {
  height:1px;
  background:linear-gradient(90deg,transparent,rgba(99,102,241,.35),transparent);
  margin:20px 0 28px 0;
}

/* Mode badge */
.mode-pill { display:inline-flex;align-items:center;gap:6px;font-size:12px;padding:4px 12px;border-radius:99px;font-weight:600;letter-spacing:.03em; }
.pill-rag     { background:rgba(34,197,94,.1);  color:#4ade80!important; border:1px solid rgba(34,197,94,.25); }
.pill-general { background:rgba(99,102,241,.1); color:#a5b4fc!important; border:1px solid rgba(99,102,241,.3); }
.pill-cd      { background:rgba(244,63,94,.1);  color:#fb7185!important; border:1px solid rgba(244,63,94,.25); animation:pulse 1.5s infinite; }

/* Upload zone */
.upload-zone-active {
  background:rgba(99,102,241,.06);
  border:1.5px solid var(--indigo);
  border-radius:16px; padding:20px; margin-bottom:16px;
  animation:fadeUp .4s ease;
}
.upload-success {
  display:flex; align-items:center; gap:12px;
  background:rgba(34,197,94,.08); border:1px solid rgba(34,197,94,.3);
  border-radius:14px; padding:14px 20px; margin-bottom:16px;
  animation:slideIn .4s ease;
}
.upload-check {
  width:28px;height:28px;border-radius:50%;
  background:linear-gradient(135deg,#22c55e,#4ade80);
  display:flex;align-items:center;justify-content:center;
  font-size:14px; flex-shrink:0;
  animation:checkPop .5s cubic-bezier(.36,.07,.19,.97) both;
}
.upload-filename { font-weight:700; color:#4ade80!important; font-size:14px; }
.upload-meta     { font-size:12px; color:#64748b!important; margin-top:2px; }

/* Key-point suggestion chips */
.suggestions-wrap { margin:0 0 20px 0; animation:slideIn .5s ease; }
.suggestions-label { font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#475569!important;margin-bottom:10px;font-weight:600; }
.chip-grid { display:flex; flex-wrap:wrap; gap:8px; }
.chip {
  display:inline-flex; align-items:center; gap:6px;
  background:rgba(99,102,241,.08); border:1px solid rgba(99,102,241,.25);
  border-radius:99px; padding:7px 16px; font-size:13px; color:#c4b5fd!important;
  cursor:pointer; transition:all .2s ease;
  animation:popIn .35s cubic-bezier(.36,.07,.19,.97) both;
}
.chip:hover { background:rgba(99,102,241,.18); border-color:var(--indigo); transform:translateY(-2px); box-shadow:0 6px 20px rgba(99,102,241,.3); }

/* Autocomplete suggestion bar */
.autocomplete-bar { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px; animation:fadeUp .3s ease; }
.ac-chip {
  background:rgba(255,255,255,.04); border:1px solid var(--glass-b);
  border-radius:6px; padding:4px 12px; font-size:12.5px; color:#94a3b8!important;
  cursor:pointer; transition:all .15s;
}
.ac-chip:hover { background:rgba(99,102,241,.12); color:#a5b4fc!important; border-color:rgba(99,102,241,.3); }
.ac-chip strong { color:#c4b5fd!important; font-weight:700; }

/* Source card */
.source-card {
  background:var(--glass); border:1px solid var(--glass-b);
  border-radius:12px; padding:14px 18px; margin-bottom:10px;
  transition:border-color .2s; animation:fadeUp .3s ease;
}
.source-card:hover { border-color:rgba(99,102,241,.4); }
.source-title { font-weight:700; font-size:13px; color:#e2e8f0!important; margin-bottom:6px; }
.source-score { background:rgba(99,102,241,.1); color:#a5b4fc!important; border:1px solid rgba(99,102,241,.25); border-radius:99px; padding:2px 10px; font-size:11px; }
.source-text  { font-size:13.5px; color:#94a3b8!important; line-height:1.65; }

/* Tip box */
.tip-box {
  background:rgba(99,102,241,.06); border:1px solid rgba(99,102,241,.18);
  border-radius:12px; padding:14px 18px; font-size:13.5px;
  color:#94a3b8!important; margin-bottom:16px; line-height:1.6;
}
.tip-box strong { color:#a5b4fc!important; }

/* Cooldown card */
.cooldown-card {
  background:rgba(244,63,94,.06); border:1px solid rgba(244,63,94,.2);
  border-radius:20px; padding:48px 32px; text-align:center; animation:fadeUp .5s ease;
}

/* Session timer bar (top of page) */
</style>
""", unsafe_allow_html=True)


# ─── Session-state bootstrap ──────────────────────────────────────────────────
def _init():
    params = st.query_params
    sid    = params.get("sid", "")

    if "session_id" not in st.session_state:
        if sid and session_exists(sid):
            st.session_state.session_id   = sid
            st.session_state.chat_history = load_session(sid)
            st.session_state.returning    = True
        else:
            st.session_state.session_id   = new_session_id()
            st.session_state.chat_history = []
            st.session_state.returning    = False

        st.session_state.session_start  = time.time()
        st.session_state.cooldown_start = None
        st.session_state.user_index     = None
        st.session_state.user_chunks    = None
        st.session_state.user_metadata  = None
        st.session_state.user_filenames = []
        st.session_state.doc_full_text  = ""
        st.session_state.key_points     = []
        st.session_state.doc_keywords   = []
        st.session_state.show_chips     = False
        st.session_state.pending_msg    = ""
        st.session_state.draft          = ""

    st.query_params["sid"] = st.session_state.session_id

_init()


# ─── Session timing ───────────────────────────────────────────────────────────
def _status():
    now = time.time()
    if st.session_state.cooldown_start is not None:
        elapsed_cd = now - st.session_state.cooldown_start
        if elapsed_cd >= COOLDOWN_SECS:
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

def _fmt(secs):
    secs = max(0, int(secs))
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


# ─── Pipeline (cached) ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_pipeline():
    if Path("./index/index.faiss").exists():
        return RAGPipeline()
    return None


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 20px 0">
      <div style="font-size:22px;font-weight:900;
                  background:linear-gradient(135deg,#a5b4fc,#c4b5fd);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        ⚡ RAGAI
      </div>
      <div style="font-size:12px;color:#475569;">AI Knowledge Assistant</div>
    </div>""", unsafe_allow_html=True)

    status, rem_a, rem_c = _status()
    pct = rem_a / SESSION_SECS if status == "active" else 0
    if status == "active":
        st.progress(pct, text=f"⏱ {_fmt(rem_a)} remaining")
    else:
        st.markdown(f'<span class="mode-pill pill-cd">🕐 Cooldown — {_fmt(rem_c)}</span>',
                    unsafe_allow_html=True)

    # Session ID Copy highlight
    st.markdown(f"""
    <div style="background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.2);
                border-radius:8px;padding:8px 12px;margin-bottom:8px;display:flex;align-items:center;">
      <span style="font-size:12px;color:#94a3b8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
        ID: <strong style="color:#a5b4fc;font-family:monospace;">{st.session_state.session_id}</strong>
      </span>
      <button class="chat-action-btn" style="margin-left:auto;font-size:12px;padding:2px 8px;border-radius:4px;background:rgba(255,255,255,0.05);" onclick="navigator.clipboard.writeText('{st.session_state.session_id}'); this.innerText='✓'; setTimeout(()=>this.innerText='Copy', 1500)">Copy</button>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Bookmark this URL to return to your conversation.")

    st.markdown("<hr style='border-color:rgba(255,255,255,.07);margin:14px 0'>",
                unsafe_allow_html=True)

    # API key override
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        st.markdown("**🔑 Groq API Key**")
        k = st.text_input("", type="password", placeholder="gsk_...",
                          key="user_api_key_field", label_visibility="collapsed")
        if k:
            os.environ["GROQ_API_KEY"] = k
        st.caption("[Get a free key →](https://console.groq.com)")
        st.markdown("<hr style='border-color:rgba(255,255,255,.07);margin:14px 0'>",
                    unsafe_allow_html=True)

    if st.session_state.user_filenames:
        st.markdown("**📄 Active documents**")
        for fn in st.session_state.user_filenames:
            st.markdown(
                f'<div style="background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.2);'
                f'border-radius:8px;padding:6px 12px;font-size:12.5px;color:#4ade80;margin-bottom:6px;">'
                f'📎 {fn}</div>',
                unsafe_allow_html=True,
            )
        if st.button("🗑 Remove documents", use_container_width=True):
            st.session_state.user_index     = None
            st.session_state.user_chunks    = None
            st.session_state.user_metadata  = None
            st.session_state.user_filenames = []
            st.session_state.doc_full_text  = ""
            st.session_state.key_points     = []
            st.session_state.doc_keywords   = []
            st.session_state.show_chips     = False
            st.rerun()
        st.markdown("<hr style='border-color:rgba(255,255,255,.07);margin:14px 0'>",
                    unsafe_allow_html=True)

    if st.button("🗂 Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        save_session(st.session_state.session_id, [])
        st.rerun()

    st.markdown("**🤖 Conversation Style**")
    style_col1, style_col2 = st.columns(2)
    with style_col1:
        is_prof = st.session_state.get("ai_mode", "professional") == "professional"
        prof_btn_style = "🟢 Professional" if is_prof else "💼 Professional"
        if st.button(prof_btn_style, key="sidebar_prof_btn", use_container_width=True):
            st.session_state.ai_mode = "professional"
            st.rerun()
    with style_col2:
        is_genz = st.session_state.get("ai_mode", "professional") == "genz"
        genz_btn_style = "🟢 Gen Z / Alpha" if is_genz else "⚡ Gen Z / Alpha"
        if st.button(genz_btn_style, key="sidebar_genz_btn", use_container_width=True):
            st.session_state.ai_mode = "genz"
            st.rerun()

    st.markdown("<hr style='border-color:rgba(255,255,255,.07);margin:14px 0'>",
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:12px;color:#475569;line-height:1.8">
      <b style="color:#64748b">Session:</b> 4 hrs active / 1 hr break<br>
      <b style="color:#64748b">Memory:</b> Per-session history
    </div>""", unsafe_allow_html=True)


# ─── Top progress bar ─────────────────────────────────────────────────────────
pct_px   = int(pct * 100)
bar_col  = "#22c55e" if pct > 0.4 else ("#f59e0b" if pct > 0.15 else "#f43f5e")
components.html(f"""
<div style="position:fixed;top:0;left:0;right:0;z-index:10000;height:3px;
            background:rgba(255,255,255,.04);">
  <div style="height:100%;width:{pct_px}%;
              background:linear-gradient(90deg,{bar_col},{bar_col}88);
              border-radius:0 99px 99px 0;transition:width 1s;"></div>
</div>""", height=0)


# ─── Load pipeline ─────────────────────────────────────────────────────────────
with st.spinner("Loading AI models…"):
    pipeline = _load_pipeline()

if pipeline is None:
    with st.spinner("Building document index from sample data…"):
        try:
            ingest.build_index("./data", "./index")
            st.rerun()
        except Exception as e:
            st.error(f"Could not build index: {e}")
            st.stop()


# ─── Cooldown gate ─────────────────────────────────────────────────────────────
status, rem_a, rem_c = _status()
if status == "cooldown":
    h = int(rem_c // 3600)
    m = int((rem_c % 3600) // 60)
    s = int(rem_c % 60)
    st.markdown(f"""
    <div class="cooldown-card">
      <div style="font-size:52px;margin-bottom:16px">☕</div>
      <div style="font-size:1.8rem;font-weight:800;color:#e2e8f0;margin-bottom:8px">Take a break!</div>
      <div style="font-size:1.05rem;color:#64748b;margin-bottom:24px">
        Your 4-hour session is complete. Rest for 1 hour, then your
        conversation continues right where you left off.
      </div>
      <div style="font-size:2.4rem;font-weight:900;font-variant-numeric:tabular-nums;
                  background:linear-gradient(135deg,#a5b4fc,#c4b5fd);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent">
        {h:02d}:{m:02d}:{s:02d}
      </div>
      <div style="font-size:13px;color:#475569;margin-top:8px">until your session resumes</div>
    </div>""", unsafe_allow_html=True)
    components.html('<script>setTimeout(()=>window.parent.location.reload(),60000);</script>', height=0)
    st.stop()


# ─── Top Middle Screen Header (RAGAI) & Timer ───────────────────────────────
timer_col = "#22c55e" if pct > 0.4 else ("#f59e0b" if pct > 0.15 else "#f43f5e")
target_secs = int(rem_a if status == "active" else rem_c)

st.markdown(f"""
<div style="text-align:center;margin-top:-15px;margin-bottom:12px;animation:fadeUp .5s ease;">
  <div style="font-size:36px;font-weight:900;letter-spacing:0.08em;
              background:linear-gradient(135deg,#c4b5fd 0%,#a5b4fc 50%,#818cf8 100%);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              text-transform:uppercase;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto;
              text-shadow: 0 0 25px rgba(99, 102, 241, 0.15);">
    ⚡ RAGAI
  </div>
</div>
""", unsafe_allow_html=True)

# Client-side countdown container with progress bar below
prefix_label = "Active" if status == "active" else "Cooldown"
components.html(f"""
<div style="text-align:center;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto;font-size:13.5px;color:#a5b4fc;font-weight:600;margin-bottom:6px;">
  ⏱ <span id="countdown-text">Calculating...</span>
</div>
<div style="width:300px;height:4px;background:rgba(255,255,255,.05);border-radius:10px;margin:0 auto 0 auto;overflow:hidden;">
  <div id="countdown-bar" style="width:100%;height:100%;background:{timer_col};border-radius:10px;transition:width 1s linear;"></div>
</div>
<script>
  let secondsLeft = {target_secs};
  const totalDuration = {SESSION_SECS if status == "active" else COOLDOWN_SECS};
  const textEl = document.getElementById("countdown-text");
  const barEl = document.getElementById("countdown-bar");
  
  function refreshTimer() {{
    if (secondsLeft <= 0) {{
      window.parent.location.reload();
      return;
    }}
    let h = Math.floor(secondsLeft / 3600);
    let rem = secondsLeft % 3600;
    let m = Math.floor(rem / 60);
    let s = rem % 60;
    let display = "";
    if (h > 0) {{
      display = h + "h " + (m < 10 ? "0" : "") + m + "m " + (s < 10 ? "0" : "") + s + "s";
    }} else {{
      display = m + "m " + (s < 10 ? "0" : "") + s + "s";
    }}
    textEl.innerText = "{prefix_label} Session: " + display;
    let percentage = (secondsLeft / totalDuration) * 100;
    barEl.style.width = percentage + "%";
    secondsLeft--;
  }}
  refreshTimer();
  setInterval(refreshTimer, 1000);
</script>
""", height=42)

using_docs = st.session_state.user_index is not None
api_key_ok = bool(os.environ.get("GROQ_API_KEY"))

mode_label = "📄 Document Mode" if using_docs else "🌐 General AI Mode"
mode_class = "pill-rag"        if using_docs else "pill-general"

style_label = "⚡ Gen Z / Alpha Style" if st.session_state.ai_mode == "genz" else "💼 Professional Style"
style_class = "pill-general"           if st.session_state.ai_mode == "genz" else "pill-rag"

st.markdown(f"""
<div style="display:flex;justify-content:center;gap:8px;margin-top:12px;margin-bottom:12px;">
  <span class="mode-pill {mode_class}">{mode_label}</span>
  <span class="mode-pill {style_class}">{style_label}</span>
</div>
<div class="nexus-divider" style="margin-top:10px;margin-bottom:15px;"></div>""", unsafe_allow_html=True)

# Switchable buttons for Conversation style on main page
tcol1, tcol2 = st.columns(2)
with tcol1:
    is_prof = st.session_state.get("ai_mode", "professional") == "professional"
    prof_label = "🟢 Professional Style" if is_prof else "💼 Switch to Professional"
    if st.button(prof_label, key="main_prof_btn", use_container_width=True):
        st.session_state.ai_mode = "professional"
        st.rerun()
with tcol2:
    is_genz = st.session_state.get("ai_mode", "professional") == "genz"
    genz_label = "🟢 Gen Z / Alpha Style" if is_genz else "⚡ Switch to Gen Z Style"
    if st.button(genz_label, key="main_genz_btn", use_container_width=True):
        st.session_state.ai_mode = "genz"
        st.rerun()


# ─── Returning user notice ─────────────────────────────────────────────────────
show_suggestion_chips = (
    st.session_state.show_chips
    and st.session_state.key_points
    and len(st.session_state.chat_history) < 6
)

if st.session_state.get("returning") and st.session_state.chat_history:
    st.markdown("""
    <div class="tip-box">
      <strong>👋 Welcome back!</strong> Your previous conversation has been restored.
    </div>""", unsafe_allow_html=True)
    st.session_state.returning = False
elif not st.session_state.chat_history and not show_suggestion_chips:
    st.markdown("""
    <div class="tip-box" style="text-align:center;">
      <strong>⚡ Ask anything.</strong> No upload needed — RAGAI answers from broad AI knowledge.
      Or drop files at the bottom for unified document Q&A.
    </div>""", unsafe_allow_html=True)


# ─── Chat history display ─────────────────────────────────────────────────────
import urllib.parse
from datetime import datetime

for msg in st.session_state.chat_history:
    role = msg["role"]
    avatar = "🧑" if role == "user" else "⚡"
    marker = '<span class="user-marker" style="display:none;">user</span>' if role == "user" else '<span class="assistant-marker" style="display:none;">assistant</span>'
    timestamp = msg.get("timestamp", datetime.now().strftime("%I:%M %p"))
    safe_content = urllib.parse.quote(msg["content"])
    
    actions_html = f"""
    <div class="chat-actions">
      <span>{timestamp}</span>
      <span style="margin: 0 2px;">·</span>
      <button class="chat-action-btn" onclick="navigator.clipboard.writeText(decodeURIComponent('{safe_content}')); this.innerText='Copied ✓'; setTimeout(()=>this.innerText='Copy', 2000)">Copy</button>
    </div>
    """
    
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"] + marker + actions_html, unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander("📎 Source highlights — click to inspect", expanded=False):
                for src in msg["sources"]:
                    hi_text = highlight_text(src["text"][:400], msg.get("query", ""))
                    st.markdown(f"""
                    <div class="source-card">
                      <div class="source-top" style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                        <span class="source-title">📄 {src["source"]}</span>
                        <span class="source-score">match {src["score"]:.0%}</span>
                      </div>
                      <div class="source-text">{hi_text}…</div>
                    </div>""", unsafe_allow_html=True)


# ─── Main-page UPLOAD zone (sit at the bottom, merged near input) ──────────────
if not using_docs:
    st.markdown('<div style="max-width:1100px;margin:12px auto 0 auto;padding:0 5px;">', unsafe_allow_html=True)
    uploads = st.file_uploader(
        "Drop .txt or .pdf files here to ask about them",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="main_uploader",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploads:
        docs, full_texts = [], []
        size_limit_exceeded = False
        for f in uploads:
            if f.size > 500 * 1024 * 1024:
                st.toast(f"⚠️ File '{f.name}' exceeds the 500MB limit!", icon="❌")
                size_limit_exceeded = True
                continue
            text = ingest.extract_text_from_bytes(f.getvalue(), f.name)
            if text.strip():
                docs.append({"text": text, "source": f.name})
                full_texts.append(text)

        if docs:
            ucol1, ucol2 = st.columns([1.5, 1])
            with ucol1:
                if st.button("📚 Index files & Ask RAGAI", key="index_files_bottom_btn", use_container_width=True):
                    with st.spinner("Indexing documents…"):
                        idx, chunks, meta = ingest.build_index_from_docs(docs, pipeline.embedder)
                    if idx:
                        st.session_state.user_index     = idx
                        st.session_state.user_chunks    = chunks
                        st.session_state.user_metadata  = meta
                        st.session_state.user_filenames = [f.name for f in uploads]
                        combined = "\n\n".join(full_texts)
                        st.session_state.doc_full_text  = combined
                        st.session_state.show_chips     = True
                        with st.spinner("✨ Analyzing key points…"):
                            if api_key_ok:
                                st.session_state.key_points   = extract_key_points(combined)
                                st.session_state.doc_keywords = extract_doc_keywords(combined)
                            else:
                                st.session_state.key_points   = []
                                st.session_state.doc_keywords = []
                        st.rerun()
            with ucol2:
                st.markdown(
                    f'<div style="padding:8px 0;font-size:12.5px;color:#64748b;text-align:right;">'
                    f'📎 {len(uploads)} file(s) selected</div>',
                    unsafe_allow_html=True,
                )
else:
    fnames = ", ".join(st.session_state.user_filenames)
    st.markdown(f"""
    <div class="upload-success" style="max-width:1100px;margin: 12px auto 6px auto;padding: 10px 16px;">
      <div class="upload-check" style="width:20px;height:20px;font-size:11px;">✓</div>
      <div style="display:flex;justify-content:space-between;width:100%;align-items:center;">
        <div>
          <span class="upload-filename" style="font-size:13px;">📎 {fnames}</span>
          <span class="upload-meta" style="font-size:11px;margin-left:8px;color:#64748b;">(Auto RAG Mode Active)</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


# ─── Key-point suggestion chips (rendered below history) ──────────────────────
if show_suggestion_chips:
    delays = [".1s", ".2s", ".3s", ".4s", ".5s"]
    chips_html = "".join(
        f'<div class="chip" style="animation-delay:{delays[i]}">'
        f'<span style="font-size:14px">💡</span> {pt}'
        f'</div>'
        for i, pt in enumerate(st.session_state.key_points)
    )
    st.markdown(f"""
    <div class="suggestions-wrap" style="max-width:1100px;margin: 8px auto;">
      <div class="suggestions-label">✨ Key points — click to ask</div>
      <div class="chip-grid">{chips_html}</div>
    </div>""", unsafe_allow_html=True)

    cols = st.columns(len(st.session_state.key_points))
    for i, (col, pt) in enumerate(zip(cols, st.session_state.key_points)):
        with col:
            if st.button(pt, key=f"kp_{i}", help="Click to ask this",
                         use_container_width=True):
                st.session_state.pending_msg = pt
                st.session_state.show_chips  = False
                st.rerun()


# ─── Autocomplete suggestion bar ─────────────────────────────────────────────
# We use a thin text_input to get real-time draft, then show matching keywords
draft = st.text_input(
    "draft_hidden",
    value=st.session_state.get("draft", ""),
    placeholder="",
    key="draft_input",
    label_visibility="collapsed",
)
st.session_state.draft = draft

# Show keyword suggestions if doc loaded and draft has content
doc_kws = st.session_state.doc_keywords
if draft and doc_kws:
    draft_lower = draft.lower()
    matches = [k for k in doc_kws if draft_lower in k.lower() or any(
        w in k.lower() for w in draft_lower.split() if len(w) > 2
    )][:6]
    if matches:
        st.markdown("""
        <div style="font-size:11px;color:#475569;letter-spacing:.05em;
                    text-transform:uppercase;margin-bottom:6px;">
          🔍 Related topics
        </div>""", unsafe_allow_html=True)
        ac_html = "".join(
            f'<span class="ac-chip">{k}</span>' for k in matches
        )
        st.markdown(f'<div class="autocomplete-bar">{ac_html}</div>',
                    unsafe_allow_html=True)
        # Clickable versions
        ac_cols = st.columns(min(len(matches), 6))
        for i, (col, kw) in enumerate(zip(ac_cols, matches)):
            with col:
                if st.button(kw, key=f"ac_{i}", use_container_width=True):
                    st.session_state.pending_msg = (draft + " " + kw).strip()
                    st.session_state.draft = ""
                    st.rerun()


# ─── Chat input ───────────────────────────────────────────────────────────────
placeholder = (
    "Ask a question about your document(s)…" if using_docs
    else "Ask anything — tech, science, career advice, coding…"
)
if not api_key_ok:
    placeholder = "⚠️ Add your Groq API key in the sidebar first…"

user_input = st.chat_input(placeholder, disabled=not api_key_ok)

# Handle chip/autocomplete pre-fill
if st.session_state.pending_msg and not user_input:
    user_input = st.session_state.pending_msg
    st.session_state.pending_msg = ""


# ─── Process message ───────────────────────────────────────────────────────────
# ─── Typewriter & Image Helper ────────────────────────────────────────────────
def typewriter_generator(stream):
    for chunk in stream:
        for char in chunk:
            yield char
            time.sleep(0.003)

if user_input:
    st.session_state.show_chips = False
    now_time = datetime.now().strftime("%I:%M %p")

    # Append user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
        "timestamp": now_time
    })
    
    with st.chat_message("user", avatar="🧑"):
        st.markdown(
            user_input + '<span class="user-marker" style="display:none;">user</span>' +
            f"""<div class="chat-actions">
              <span>{now_time}</span>
              <span style="margin: 0 2px;">·</span>
              <button class="chat-action-btn" onclick="navigator.clipboard.writeText(decodeURIComponent('{urllib.parse.quote(user_input)}')); this.innerText='Copied ✓'; setTimeout(()=>this.innerText='Copy', 2000)">Copy</button>
            </div>""",
            unsafe_allow_html=True
        )

    # Build history for LLM (exclude current message)
    history_for_llm = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.chat_history[:-1]
    ]

    sources = []
    mode = st.session_state.get("ai_mode", "professional")
    
    # Check if request is an image generation request
    image_keywords = ["generate image", "draw", "create image", "paint", "show me a picture of"]
    is_image_request = any(user_input.lower().startswith(x) for x in image_keywords)
    
    with st.chat_message("assistant", avatar="⚡"):
        if is_image_request:
            img_prompt = user_input
            for x in ["generate image of", "generate image", "draw a picture of", "draw", "create image of", "create image", "paint", "show me a picture of"]:
                if img_prompt.lower().startswith(x):
                    img_prompt = img_prompt[len(x):].strip()
                    break
            
            # URL encode prompt
            encoded_prompt = urllib.parse.quote(img_prompt)
            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=768&height=768&nologo=true&seed={int(time.time())}"
            
            text_desc = f"🎨 Here is the image I generated for: **{img_prompt}**"
            st.write_stream(typewriter_generator([text_desc]))
            st.image(image_url, use_column_width=True)
            
            full_response = f"🎨 Here is the image I generated for: **{img_prompt}**\n\n![Generated Image]({image_url})"
            st.markdown('<span class="assistant-marker" style="display:none;">assistant</span>' +
                        f"""<div class="chat-actions">
                          <span>{now_time}</span>
                          <span style="margin: 0 2px;">·</span>
                          <button class="chat-action-btn" onclick="navigator.clipboard.writeText(decodeURIComponent('{urllib.parse.quote(full_response)}')); this.innerText='Copied ✓'; setTimeout(()=>this.innerText='Copy', 2000)">Copy</button>
                        </div>""", unsafe_allow_html=True)
        else:
            if using_docs:
                retrieved = pipeline.retrieve(
                    user_input,
                    index=st.session_state.user_index,
                    chunks=st.session_state.user_chunks,
                    metadata=st.session_state.user_metadata,
                )
                sources = retrieved
                full_response = st.write_stream(
                    typewriter_generator(pipeline.stream_rag_answer(user_input, retrieved, history_for_llm, ai_mode=mode))
                )
            else:
                full_response = st.write_stream(
                    typewriter_generator(pipeline.stream_general_answer(user_input, history_for_llm, ai_mode=mode))
                )
                
            st.markdown('<span class="assistant-marker" style="display:none;">assistant</span>' +
                        f"""<div class="chat-actions">
                          <span>{now_time}</span>
                          <span style="margin: 0 2px;">·</span>
                          <button class="chat-action-btn" onclick="navigator.clipboard.writeText(decodeURIComponent('{urllib.parse.quote(full_response)}')); this.innerText='Copied ✓'; setTimeout(()=>this.innerText='Copy', 2000)">Copy</button>
                        </div>""", unsafe_allow_html=True)

        if sources:
            with st.expander("📎 Source highlights — click to inspect", expanded=False):
                for src in sources:
                    hi_text = highlight_text(src["text"][:400], user_input)
                    st.markdown(f"""
                    <div class="source-card">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                        <span class="source-title">📄 {src["source"]}</span>
                        <span class="source-score">match {src["score"]:.0%}</span>
                      </div>
                      <div class="source-text">{hi_text}…</div>
                    </div>""", unsafe_allow_html=True)

    # Append assistant message to history
    st.session_state.chat_history.append({
        "role":      "assistant",
        "content":   full_response,
        "sources":   sources,
        "query":     user_input,
        "timestamp": datetime.now().strftime("%I:%M %p")
    })

    # Persist
    save_session(st.session_state.session_id, st.session_state.chat_history)
    st.rerun()


st.markdown("""
<div style="text-align:center;margin-top:48px;padding-top:20px;
            border-top:1px solid rgba(255,255,255,.06);
            font-size:12.5px;color:#334155">
  RAGAI · Secure AI Knowledge Assistant
</div>""", unsafe_allow_html=True)