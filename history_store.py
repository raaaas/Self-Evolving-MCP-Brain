"""
Server-side history persistence (JSON file store).
Provides a FastAPI router mounted into the main ui_server app.

Storage: .mcp_history/history.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

HISTORY_DIR = Path(__file__).resolve().parent / ".mcp_history"
HISTORY_FILE = HISTORY_DIR / "history.json"

router = APIRouter(prefix="/history", tags=["history"])


def _ensure_dir() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _read_all() -> list[dict[str, Any]]:
    _ensure_dir()
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _write_all(sessions: list[dict[str, Any]]) -> None:
    _ensure_dir()
    tmp = HISTORY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, HISTORY_FILE)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class HistorySession(BaseModel):
    id: int
    snippet: str = ""
    crystallized: bool = False
    timestamp: str = ""
    snapshot: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.get("", response_model=list[HistorySession])
def list_sessions():
    """Return all sessions ordered newest-first (caller manages order)."""
    return _read_all()


@router.post("", response_model=HistorySession)
def upsert_session(session: HistorySession):
    """Create or update a session. Matched by `id`."""
    sessions = _read_all()
    existing = next((s for s in sessions if s.get("id") == session.id), None)
    if existing is not None:
        existing.update(session.model_dump())
    else:
        sessions.insert(0, session.model_dump())
    _write_all(sessions)
    return session


@router.delete("/{session_id}")
def delete_session(session_id: int):
    """Remove a single session by its numeric id."""
    sessions = _read_all()
    before = len(sessions)
    sessions = [s for s in sessions if s.get("id") != session_id]
    if len(sessions) == before:
        raise HTTPException(status_code=404, detail="Session not found")
    _write_all(sessions)
    return {"deleted": session_id}
