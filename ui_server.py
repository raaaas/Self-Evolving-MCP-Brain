"""
Self-Evolving MCP Brain — UI backend (FastAPI on port 8000).

PHASE 2 scope:
  - HTTP API for the conversational analyzer UI.
  - Bridges user input (URLs, GitHub/CodePen links, raw code, image URLs) to the
    local freellmapi proxy (raw HTTP, OpenAI wire format).
  - Provides iterative /chat so the LLM can ask clarifying questions across turns.

Deliberate constraints:
  - NO `openai` SDK; raw `requests` only (per project decision).
  - NO mock data. If freellmapi is unreachable or FREELLMAPI_KEY is unset, every
    LLM-touching endpoint returns a clear structured error.
  - The system prompt is sent VERBATIM as specified by the human, then extended
    with a tiny "transport" note (NOT a behavioral change) so the LLM knows it
    cannot browse and must rely on fetched context.

PHASE 4.1 additions:
  - POST /upload: multipart/form-data file uploads (ZIP, images, text files).
  - ZIP extraction is secure: path traversal blocked, node_modules/.git skipped,
    binary filtered, total size capped.
  - Images encoded as base64 data-URLs for OpenAI vision content blocks.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, AsyncIterator, Literal

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from crystallize import crystallize_reply
from history_store import router as history_router
from library_store import router as library_router, extract_tags_via_llm, add_entry
from freellmapi_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    FreeLLMAPIError,
    FreeLLMAPITimeout,
    chat as llm_chat,
    list_models as llm_list_models,
)
from upload_handler import process_upload

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FRONTEND_ORIGIN = "http://localhost:3000"
SYSTEM_PROMPT = (
    "Analyze this design/code input. Extract the core architectural patterns, "
    "UI/UX rules, and coding standards. Ask clarifying questions if the input "
    "is ambiguous. Generate a Vue.js/GSAP/Three.js sample code demonstrating "
    "this pattern."
)
# Transport note — clarifies capability, does NOT alter the behavioral instruction.
TRANSPORT_NOTE = (
    "\n\nOperational note: you have no internet access. Any URL content has been "
    "fetched for you and is included inline. Do not pretend to have browsed. "
    "Image URLs are delivered via vision; video URLs cannot be watched — for "
    "video, ask the user to describe the relevant part."
)

# Detection patterns for input classification
IMAGE_EXT_RE = re.compile(r"\.(png|jpe?g|gif|webp|svg|bmp)(\?.*)?$", re.IGNORECASE)
RAW_GITHUB_RE = re.compile(r"^https?://raw\.githubusercontent\.com/.+")
GITHUB_RE = re.compile(r"^https?://github\.com/.+")
CODEPEN_RE = re.compile(r"^https?://(codepen\.io|cp\.tt)/.+")
HTTP_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

FETCH_TIMEOUT = 20  # seconds for server-side URL fetching
MAX_FETCH_BYTES = 200_000  # cap fetched text so we don't blow the context window

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Self-Evolving Brain — UI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST", "OPTIONS", "DELETE"],
    allow_headers=["*"],
)

app.include_router(history_router)
app.include_router(library_router)


# ---------------------------------------------------------------------------
# Pydantic models (pydantic v2 — uses model_dump())
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    input: str = Field(..., description="A URL, GitHub/CodePen link, raw code, or image URL")
    model: str | None = None
    stream: bool = False


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    history: list[ChatMessage] = Field(..., description="Full prior conversation, oldest-first")
    message: str
    model: str | None = None
    stream: bool = False


class CrystallizeRequest(BaseModel):
    """The approved LLM response to persist as a skill.

    The client may send the raw reply (and we extract rules + code server-side,
    matching the frontend's preview logic) OR send already-extracted rules/code
    if it has them. `skill_name` is optional; we derive a default if absent.
    """
    reply: str = Field(..., description="The approved assistant reply text")
    skill_name: str | None = None
    rules: str | None = None
    code_blocks: list[dict[str, str]] | None = None  # [{lang, code}, ...]
    session_id: int | None = None


class AnalyzeResponse(BaseModel):
    reply: str
    model: str
    routed_via: str | None
    input_type: str  # what we classified the input as, for transparency in the UI


class ModelsResponse(BaseModel):
    models: list[str]
    default: str


class CrystallizeResponse(BaseModel):
    status: str
    filename: str
    skill_name: str
    saved_path: str
    rules_chars: int
    code_block_count: int
    available_to_mcp: bool
    tags: list[str] = []
    category: str = ""
    library_entry_added: bool = False


class HealthResponse(BaseModel):
    status: str
    freellmapi_base_url: str
    freellmapi_reachable: bool
    key_configured: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# Input classification + URL fetching
# ---------------------------------------------------------------------------
def classify_input(raw: str) -> str:
    """Return one of: 'image_url', 'video_url', 'github', 'codepen', 'web_url', 'raw_code'."""
    s = raw.strip()
    if not s:
        return "raw_code"
    if IMAGE_EXT_RE.search(s):
        return "image_url"
    if re.search(r"\.(mp4|webm|mov|m3u8|ogg)(\?.*)?$", s, re.IGNORECASE):
        return "video_url"
    if RAW_GITHUB_RE.match(s):
        return "github"
    if GITHUB_RE.match(s):
        return "github"
    if CODEPEN_RE.match(s):
        return "codepen"
    if HTTP_URL_RE.match(s):
        return "web_url"
    return "raw_code"


def _fetch_text(url: str) -> tuple[bool, str]:
    """Fetch a URL as text. Returns (ok, summary). Never raises — fails soft."""
    try:
        resp = requests.get(
            url,
            timeout=FETCH_TIMEOUT,
            headers={"User-Agent": "self-evolving-brain/0.1"},
        )
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()
        text = resp.text[:MAX_FETCH_BYTES]
        truncated = len(resp.text) > MAX_FETCH_BYTES
        note = f" (truncated to {MAX_FETCH_BYTES} bytes)" if truncated else ""
        if "html" in ctype:
            return True, f"[fetched HTML, {len(text)} bytes{note}]\n{text}"
        return True, f"[fetched {ctype or 'text'}, {len(text)} bytes{note}]\n{text}"
    except requests.RequestException as exc:
        return False, str(exc)


def build_initial_messages(user_input: str) -> tuple[list[dict[str, Any]], str]:
    """Construct the OpenAI messages list for the first analyze call.

    Returns (messages, input_type).
    """
    input_type = classify_input(user_input)
    system_msgs = [{"role": "system", "content": SYSTEM_PROMPT + TRANSPORT_NOTE}]

    if input_type == "image_url":
        # Native vision: content is a list of content blocks.
        user_content = [
            {"type": "text", "text": "Analyze this image. " + user_input},
            {"type": "image_url", "image_url": {"url": user_input}},
        ]
        return system_msgs + [{"role": "user", "content": user_content}], input_type

    if input_type in ("github", "codepen", "web_url"):
        ok, fetched = _fetch_text(user_input)
        if ok:
            body = (
                f"User submitted this URL: {user_input}\n\n{fetched}\n\n"
                "Base your analysis on the fetched content above."
            )
        else:
            body = (
                f"User submitted this URL: {user_input}\n\n"
                f"Server-side fetch failed ({fetched}). The URL likely blocks "
                "automated requests (e.g. Cloudflare on CodePen, rate-limit on "
                "GitHub, etc.).\n\n"
                "You MUST ask the user to paste the relevant code (HTML, CSS, JS) "
                "directly into the chat. Do NOT pretend you can see the page. "
                "Do NOT guess at the code."
            )
        return system_msgs + [{"role": "user", "content": body}], input_type

    if input_type == "video_url":
        body = (
            f"User submitted a video URL: {user_input}\n"
            "You cannot watch video. Ask the user to describe the relevant part "
            "of the video or paste any associated code."
        )
        return system_msgs + [{"role": "user", "content": body}], input_type

    # raw_code (the common case for pasted code)
    body = f"User submitted the following code/text to analyze:\n\n```\n{user_input}\n```"
    return system_msgs + [{"role": "user", "content": body}], input_type


# ---------------------------------------------------------------------------
# Streaming helpers (NDJSON: one JSON object per line)
# ---------------------------------------------------------------------------
def _ndjson_line(obj: dict[str, Any]) -> bytes:
    """Encode a dict as a single NDJSON line (newline-terminated)."""
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def _progress(step: str, message: str) -> bytes:
    return _ndjson_line({"type": "progress", "step": step, "message": message})


def _result(reply: str, model: str, routed_via: str | None, input_type: str) -> bytes:
    return _ndjson_line({
        "type": "result",
        "reply": reply,
        "model": model,
        "routed_via": routed_via,
        "input_type": input_type,
    })


def _error(message: str) -> bytes:
    return _ndjson_line({"type": "error", "message": message})


def _done() -> bytes:
    return _ndjson_line({"type": "done"})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
def llm_error_to_http(exc: FreeLLMAPIError) -> HTTPException:
    """Map an LLM client error to the correct HTTP status.
    Timeouts (with retry in freellmapi_client) -> 504, others -> 502.
    """
    status = 504 if isinstance(exc, FreeLLMAPITimeout) else 502
    return HTTPException(status_code=status, detail=str(exc))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report proxy reachability + key status. Never calls the LLM."""
    import os

    key_set = bool(os.getenv("FREELLMAPI_KEY"))
    reachable = False
    err: str | None = None
    try:
        # /v1/models is the cheapest reachability probe
        llm_list_models()
        reachable = True
    except FreeLLMAPIError as exc:
        err = str(exc)
    return HealthResponse(
        status="ok" if reachable else "degraded",
        freellmapi_base_url=DEFAULT_BASE_URL,
        freellmapi_reachable=reachable,
        key_configured=key_set,
        error=err,
    )


@app.get("/models", response_model=ModelsResponse)
async def models() -> ModelsResponse:
    try:
        m = await run_in_threadpool(llm_list_models)
        return ModelsResponse(models=m, default=DEFAULT_MODEL)
    except FreeLLMAPIError as exc:
        raise llm_error_to_http(exc) from exc


@app.post("/analyze_input", response_model=AnalyzeResponse)
async def analyze_input(req: AnalyzeRequest) -> AnalyzeResponse:
    """First turn: classify + assemble context, then ask the LLM to analyze."""
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="`input` must not be empty.")

    if req.stream:
        return StreamingResponse(
            _stream_analyze(req),
            media_type="application/x-ndjson",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    messages, input_type = build_initial_messages(req.input)
    try:
        result = await run_in_threadpool(llm_chat, messages, model=req.model or DEFAULT_MODEL)
    except FreeLLMAPIError as exc:
        raise llm_error_to_http(exc) from exc

    return AnalyzeResponse(
        reply=result.content,
        model=result.model,
        routed_via=result.routed_via,
        input_type=input_type,
    )


async def _stream_analyze(req: AnalyzeRequest) -> AsyncIterator[bytes]:
    """Yield NDJSON progress events + final result for an analyze request."""
    raw = req.input.strip()

    # Step 1 — classify
    yield _progress("classify", "Classifying input…")
    input_type = classify_input(raw)

    # Step 2 — fetch if URL (do it here so we can yield progress)
    fetch_ok = False
    fetch_result = ""
    if input_type in ("github", "codepen", "web_url"):
        yield _progress("fetch", f"Fetching {raw[:60]}…")
        fetch_ok, fetch_result = await run_in_threadpool(_fetch_text, raw)
        yield _progress("build", "Building prompt…")
    elif input_type == "image_url":
        yield _progress("build", "Building vision prompt…")
    elif input_type == "video_url":
        yield _progress("build", "Building prompt…")
    else:
        yield _progress("build", "Building prompt…")

    # Build messages (pass pre-fetched content to avoid double-fetch)
    messages = _build_messages_from_parts(raw, input_type, fetch_ok, fetch_result)

    # Step 3 — call LLM
    yield _progress("llm", "Analyzing with model…")
    try:
        result = await run_in_threadpool(llm_chat, messages, model=req.model or DEFAULT_MODEL)
    except FreeLLMAPIError as exc:
        yield _error(str(exc))
        yield _done()
        return

    yield _result(result.content, result.model, result.routed_via, input_type)
    yield _done()


def _build_messages_from_parts(
    raw: str, input_type: str, fetch_ok: bool, fetch_result: str
) -> list[dict[str, Any]]:
    """Build messages list from pre-classified input + optional pre-fetched body."""
    system_msgs = [{"role": "system", "content": SYSTEM_PROMPT + TRANSPORT_NOTE}]

    if input_type == "image_url":
        user_content = [
            {"type": "text", "text": "Analyze this image. " + raw},
            {"type": "image_url", "image_url": {"url": raw}},
        ]
        return system_msgs + [{"role": "user", "content": user_content}]

    if input_type in ("github", "codepen", "web_url"):
        if fetch_ok:
            body = (
                f"User submitted this URL: {raw}\n\n{fetch_result}\n\n"
                "Base your analysis on the fetched content above."
            )
        else:
            body = (
                f"User submitted this URL: {raw}\n\n"
                f"Server-side fetch failed ({fetch_result}). The URL likely blocks "
                "automated requests (e.g. Cloudflare on CodePen, rate-limit on "
                "GitHub, etc.).\n\n"
                "You MUST ask the user to paste the relevant code (HTML, CSS, JS) "
                "directly into the chat. Do NOT pretend you can see the page. "
                "Do NOT guess at the code."
            )
        return system_msgs + [{"role": "user", "content": body}]

    if input_type == "video_url":
        body = (
            f"User submitted a video URL: {raw}\n"
            "You cannot watch video. Ask the user to describe the relevant part "
            "of the video or paste any associated code."
        )
        return system_msgs + [{"role": "user", "content": body}]

    # raw_code
    body = f"User submitted the following code/text to analyze:\n\n```\n{raw}\n```"
    return system_msgs + [{"role": "user", "content": body}]


@app.post("/upload", response_model=AnalyzeResponse)
async def upload(
    file: UploadFile = File(...),
    model: str = Form(None),
    stream: bool = Form(False),
) -> AnalyzeResponse:
    """Accept a file upload (ZIP, image, or text/code file) and analyze it.

    multipart/form-data with fields: file (required), model (optional), stream (optional).
    When stream=true the response is NDJSON progress events + final result.
    """
    # FIRST-LINE log: visible before any processing, so a hang is diagnosable.
    print(f"[upload] ENTER filename={file.filename!r} content_type={file.content_type!r}",
          flush=True)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    file_bytes = await file.read()
    print(f"[upload] read {len(file_bytes)} bytes", flush=True)
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if stream:
        return StreamingResponse(
            _stream_upload(file_bytes, file.filename, model),
            media_type="application/x-ndjson",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    # Non-streaming path (original)
    try:
        upload_result = await run_in_threadpool(process_upload, file_bytes, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    print(f"[upload] processed type={upload_result.upload_type} summary={upload_result.summary!r}",
          flush=True)

    system_msgs = [{"role": "system", "content": SYSTEM_PROMPT + TRANSPORT_NOTE}]

    if upload_result.upload_type == "image":
        user_msg = {"role": "user", "content": upload_result.content}
    else:
        body = (
            f"User uploaded a file: {upload_result.filename}\n"
            f"Summary: {upload_result.summary}\n\n"
            f"```\n{upload_result.content}\n```"
        )
        user_msg = {"role": "user", "content": body}

    messages = system_msgs + [user_msg]
    print(f"[upload] calling LLM (messages={len(messages)})", flush=True)

    try:
        result = await run_in_threadpool(llm_chat, messages, model=model or DEFAULT_MODEL)
    except FreeLLMAPIError as exc:
        raise llm_error_to_http(exc) from exc

    print(f"[upload] LLM done model={result.model}", flush=True)
    return AnalyzeResponse(
        reply=result.content,
        model=result.model,
        routed_via=result.routed_via,
        input_type=upload_result.upload_type,
    )


async def _stream_upload(file_bytes: bytes, filename: str, model: str | None) -> AsyncIterator[bytes]:
    """Yield NDJSON progress events + final result for a file upload."""
    yield _progress("reading", f"Reading {filename} ({len(file_bytes)} bytes)…")

    yield _progress("processing", "Processing file…")
    try:
        upload_result = await run_in_threadpool(process_upload, file_bytes, filename)
    except ValueError as exc:
        yield _error(str(exc))
        yield _done()
        return

    print(f"[upload] processed type={upload_result.upload_type} summary={upload_result.summary!r}",
          flush=True)

    system_msgs = [{"role": "system", "content": SYSTEM_PROMPT + TRANSPORT_NOTE}]

    if upload_result.upload_type == "image":
        user_msg = {"role": "user", "content": upload_result.content}
    else:
        body = (
            f"User uploaded a file: {upload_result.filename}\n"
            f"Summary: {upload_result.summary}\n\n"
            f"```\n{upload_result.content}\n```"
        )
        user_msg = {"role": "user", "content": body}

    messages = system_msgs + [user_msg]
    print(f"[upload] calling LLM (messages={len(messages)})", flush=True)

    yield _progress("llm", "Analyzing with model…")
    try:
        result = await run_in_threadpool(llm_chat, messages, model=model or DEFAULT_MODEL)
    except FreeLLMAPIError as exc:
        yield _error(str(exc))
        yield _done()
        return

    print(f"[upload] LLM done model={result.model}", flush=True)
    yield _result(result.content, result.model, result.routed_via, upload_result.upload_type)
    yield _done()


@app.post("/chat", response_model=AnalyzeResponse)
async def chat(req: ChatRequest) -> AnalyzeResponse:
    """Subsequent turns. History is the full prior conversation; we append the
    new user message and send it all to the LLM (stateless on our side)."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="`message` must not be empty.")

    if req.stream:
        return StreamingResponse(
            _stream_chat(req),
            media_type="application/x-ndjson",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    messages: list[dict[str, Any]] = [{"role": m.role, "content": m.content} for m in req.history]
    messages.append({"role": "user", "content": req.message})

    try:
        result = await run_in_threadpool(llm_chat, messages, model=req.model or DEFAULT_MODEL)
    except FreeLLMAPIError as exc:
        raise llm_error_to_http(exc) from exc

    return AnalyzeResponse(
        reply=result.content,
        model=result.model,
        routed_via=result.routed_via,
        input_type="chat",
    )


async def _stream_chat(req: ChatRequest) -> AsyncIterator[bytes]:
    """Yield NDJSON progress events + final result for a chat continuation."""
    yield _progress("llm", "Analyzing with model…")

    messages: list[dict[str, Any]] = [{"role": m.role, "content": m.content} for m in req.history]
    messages.append({"role": "user", "content": req.message})

    try:
        result = await run_in_threadpool(llm_chat, messages, model=req.model or DEFAULT_MODEL)
    except FreeLLMAPIError as exc:
        yield _error(str(exc))
        yield _done()
        return

    yield _result(result.content, result.model, result.routed_via, "chat")
    yield _done()


@app.post("/crystallize", response_model=CrystallizeResponse)
def crystallize(req: CrystallizeRequest) -> CrystallizeResponse:
    """Persist the approved response as a structured Markdown skill.

    The file is written atomically into .mcp_skills/. Because the MCP resource
    `skills://rhythm-standards` reads that directory fresh on every call, the
    new skill is immediately available to agents — no restart required.

    Extraction rules:
      * If the client sends pre-extracted `rules` and `code_blocks`, those win.
      * Otherwise we extract from `reply` using the same logic the frontend
        preview uses (so what the user saw == what gets saved).
    """
    if not req.reply.strip():
        raise HTTPException(status_code=400, detail="`reply` must not be empty.")

    # If the client provided already-extracted pieces, prefer them (the user may
    # have edited the rules or pruned code blocks before approving).
    if req.rules is not None or req.code_blocks is not None:
        from crystallize import save_skill
        rules_text = (req.rules or "").strip()
        blocks = [
            ((b.get("lang") or "").lower(), b.get("code", ""))
            for b in (req.code_blocks or [])
            if b.get("code", "").strip()
        ]
        result = save_skill(
            skill_name=(req.skill_name or "").strip() or "untitled-skill",
            rules_text=rules_text,
            code_blocks=blocks,
        )
    else:
        result = crystallize_reply(
            reply=req.reply,
            skill_name=req.skill_name,
        )

    # Confirm the file is now visible to the MCP resource by re-reading the dir.
    from crystallize import SKILLS_DIR
    available = result.saved_path in set(SKILLS_DIR.iterdir())

    tags: list[str] = []
    category = "Uncategorized"
    lib_added = False
    try:
        tags, category = extract_tags_via_llm(
            skill_name=result.skill_name,
            rules_text=result.rules_text,
            code_block_count=len(result.code_blocks),
        )
    except Exception:
        pass

    try:
        from datetime import datetime
        add_entry({
            "filename": result.filename,
            "name": result.skill_name,
            "tags": tags,
            "category": category,
            "crystallized_at": datetime.now().isoformat(),
            "session_id": req.session_id,
            "rule_chars": len(result.rules_text),
            "code_block_count": len(result.code_blocks),
            "forked_from": None,
        })
        lib_added = True
    except Exception:
        pass

    return CrystallizeResponse(
        status="crystallized",
        filename=result.filename,
        skill_name=result.skill_name,
        saved_path=str(result.saved_path),
        rules_chars=len(result.rules_text),
        code_block_count=len(result.code_blocks),
        available_to_mcp=available,
        tags=tags,
        category=category,
        library_entry_added=lib_added,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("ui_server:app", host="0.0.0.0", port=8000, reload=False)
