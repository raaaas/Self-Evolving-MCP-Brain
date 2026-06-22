"""
Self-Evolving MCP Brain — static frontend server (port 3000).

Serves the single-page Vue 3 frontend from ./frontend/.
Tiny purpose-built static server — no build step, Vue loaded from CDN.
"""

from __future__ import annotations

import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
API_BASE_URL = "http://localhost:8000"  # the UI backend


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Modern lifespan handler (replaces the deprecated @app.on_event)."""
    # Small convenience: log where the frontend will point.
    print(f"[frontend] serving {FRONTEND_DIR}  ->  API at {API_BASE_URL}")
    yield
    # No shutdown work needed for this static server.


app = FastAPI(title="Self-Evolving Brain — Frontend", lifespan=lifespan)

# The page talks to :8000 (different origin) — allow that origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # static file server; the real CORS is enforced by :8000
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    idx = FRONTEND_DIR / "index.html"
    if not idx.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html missing")
    return FileResponse(idx, media_type="text/html")


@app.get("/{path:path}")
def static(path: str):
    # Prevent path traversal
    target = (FRONTEND_DIR / path).resolve()
    try:
        target.relative_to(FRONTEND_DIR)
    except ValueError:
        raise HTTPException(status_code=404)
    if not target.is_file():
        raise HTTPException(status_code=404)
    mime, _ = mimetypes.guess_type(target.name)
    return FileResponse(target, media_type=mime or "application/octet-stream")


if __name__ == "__main__":
    uvicorn.run("frontend_server:app", host="0.0.0.0", port=3000, reload=False)
