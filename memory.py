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
    path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "updated_at": datetime.now().isoformat(),
                "history": history,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def load_session(session_id: str) -> list:
    """Return the chat history list for a session, or [] if not found."""
    path = MEMORY_DIR / f"{session_id}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("history", [])
        except Exception:
            pass
    return []


def session_exists(session_id: str) -> bool:
    return (MEMORY_DIR / f"{session_id}.json").exists()
