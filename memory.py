"""
memory.py — Session memory persistence for NexusAI.
Saves and loads per-user chat history keyed by a short session ID.
Files live in memory/ (gitignored) so they are never pushed to GitHub.
"""
import json
import uuid
from pathlib import Path
from datetime import datetime

MEMORY_DIR = Path("memory")


def _ensure_dir():
    MEMORY_DIR.mkdir(exist_ok=True)


def new_session_id() -> str:
    """Generate a short, URL-safe session identifier."""
    return str(uuid.uuid4()).replace("-", "")[:12]


def save_session(session_id: str, history: list) -> None:
    """Persist the chat history list for the given session ID."""
    _ensure_dir()
    path = MEMORY_DIR / f"{session_id}.json"
    # Sanitize history to remove any non-serializable chars
    safe_history = []
    for msg in history:
        safe_msg = {k: (v.encode('utf-8', errors='replace').decode('utf-8') if isinstance(v, str) else v)
                    for k, v in msg.items()}
        safe_history.append(safe_msg)
    with open(path, 'w', encoding='utf-8', errors='replace') as f:
        json.dump(
            {
                "session_id": session_id,
                "updated_at": datetime.now().isoformat(),
                "history": safe_history,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def load_session(session_id: str) -> list:
    """Return the chat history list for a session, or [] if not found."""
    path = MEMORY_DIR / f"{session_id}.json"
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                data = json.load(f)
            return data.get("history", [])
        except Exception:
            pass
    return []


def session_exists(session_id: str) -> bool:
    return (MEMORY_DIR / f"{session_id}.json").exists()
