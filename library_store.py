"""
Library metadata index for crystallized skills.

Stores a JSON array at .mcp_skills/library_index.json mapping skill filenames
to searchable metadata (tags, category, session_id, etc.). Provides a FastAPI
router mounted at /library in ui_server.py.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from crystallize import (
    SKILLS_DIR,
    CrystallizeResult,
    extract_rules,
    extract_code_blocks,
    save_skill,
    build_markdown,
    make_filename,
)

PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_FILE = SKILLS_DIR / "library_index.json"

router = APIRouter(prefix="/library", tags=["library"])


# ---------------------------------------------------------------------------
# Index I/O
# ---------------------------------------------------------------------------
def read_index() -> list[dict[str, Any]]:
    if not INDEX_FILE.exists():
        return []
    try:
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def write_index(entries: list[dict[str, Any]]) -> None:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_FILE.with_suffix(INDEX_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, INDEX_FILE)


def add_entry(entry: dict[str, Any]) -> None:
    entries = read_index()
    entries.insert(0, entry)
    write_index(entries)


def update_entry(filename: str, patch: dict[str, Any]) -> dict[str, Any] | None:
    entries = read_index()
    for e in entries:
        if e.get("filename") == filename:
            e.update(patch)
            write_index(entries)
            return e
    return None


def remove_entry(filename: str) -> bool:
    entries = read_index()
    before = len(entries)
    entries = [e for e in entries if e.get("filename") != filename]
    if len(entries) == before:
        return False
    write_index(entries)
    skill_file = SKILLS_DIR / filename
    if skill_file.exists():
        skill_file.unlink()
    return True


def rebuild_index() -> int:
    """Scan .mcp_skills/*.md and create entries for files missing from index."""
    if not SKILLS_DIR.exists():
        return 0
    entries = read_index()
    indexed = {e["filename"] for e in entries}
    added = 0
    for md_file in sorted(SKILLS_DIR.glob("skill_*.md")):
        if md_file.name in indexed:
            continue
        text = md_file.read_text(encoding="utf-8")
        rules_text = extract_rules(text)
        blocks = extract_code_blocks(text)
        name = _derive_name(md_file.name)
        entry = {
            "filename": md_file.name,
            "name": name,
            "tags": [],
            "category": "Uncategorized",
            "crystallized_at": "",
            "session_id": None,
            "rule_chars": len(rules_text),
            "code_block_count": len(blocks),
            "forked_from": None,
        }
        entries.insert(0, entry)
        added += 1
    if added:
        write_index(entries)
    return added


def _derive_name(filename: str) -> str:
    """Derive a readable name from a skill filename."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "skill":
        parts = parts[1:]
    if parts and len(parts[-1]) == 12 and parts[-1].isdigit():
        parts = parts[:-1]
    name = " ".join(parts).replace("_", " ").title()
    return name or "Untitled Skill"


# ---------------------------------------------------------------------------
# Pydantic models for library endpoints
# ---------------------------------------------------------------------------
class RegenerateRequest(BaseModel):
    model: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.get("/{filename}/content")
def get_library_content(filename: str):
    """Return the parsed content of a skill file: name, rules, code blocks."""
    entries = read_index()
    entry = next((e for e in entries if e.get("filename") == filename), None)
    skill_file = SKILLS_DIR / filename
    if not skill_file.exists():
        raise HTTPException(status_code=404, detail="Skill file not found")
    text = skill_file.read_text(encoding="utf-8")
    rules = extract_rules(text)
    blocks = extract_code_blocks(text)
    return {
        "filename": filename,
        "name": entry.get("name") if entry else filename,
        "rules": rules,
        "code_blocks": [{"lang": lang, "code": code} for lang, code in blocks],
    }


@router.get("")
def list_library():
    """Return all index entries sorted newest-first (index order)."""
    entries = read_index()
    added = rebuild_index()
    if added:
        entries = read_index()
    return entries


@router.delete("/{filename}")
def delete_library_item(filename: str):
    """Remove a skill entry and its .md file."""
    if not remove_entry(filename):
        raise HTTPException(status_code=404, detail="Library entry not found")
    return {"deleted": filename}


@router.post("/{filename}/fork")
def fork_library_item(filename: str):
    """Duplicate a skill .md file with a new timestamped filename."""
    entries = read_index()
    source_entry = next((e for e in entries if e.get("filename") == filename), None)
    if source_entry is None:
        raise HTTPException(status_code=404, detail="Source library entry not found")

    src_file = SKILLS_DIR / filename
    if not src_file.exists():
        raise HTTPException(status_code=404, detail="Source skill file not found")

    text = src_file.read_text(encoding="utf-8")
    rules_text = extract_rules(text)
    blocks = extract_code_blocks(text)
    fork_name = (source_entry.get("name") or "untitled") + " Fork"
    result = save_skill(fork_name, rules_text, blocks)

    new_entry = {
        "filename": result.filename,
        "name": fork_name,
        "tags": list(source_entry.get("tags") or []),
        "category": source_entry.get("category", "Uncategorized"),
        "crystallized_at": datetime.now().isoformat(),
        "session_id": None,
        "rule_chars": len(rules_text),
        "code_block_count": len(blocks),
        "forked_from": filename,
    }
    add_entry(new_entry)
    return new_entry


@router.post("/{filename}/regenerate")
async def regenerate_library_item(filename: str, req: RegenerateRequest):
    """Regenerate a skill via LLM using original conversation context.

    Reads the session history, sends the original conversation + current skill
    content to the LLM, re-extracts rules and code, re-tags, and overwrites
    the .md file and index entry.
    """
    from fastapi.concurrency import run_in_threadpool
    from freellmapi_client import (
        DEFAULT_MODEL,
        chat as llm_chat,
        FreeLLMAPIError,
    )
    from history_store import _read_all as read_all_history

    entries = read_index()
    entry = next((e for e in entries if e.get("filename") == filename), None)
    if entry is None:
        raise HTTPException(status_code=404, detail="Library entry not found")

    skill_file = SKILLS_DIR / filename
    if not skill_file.exists():
        raise HTTPException(status_code=404, detail="Skill file not found")

    current_text = skill_file.read_text(encoding="utf-8")
    current_rules = extract_rules(current_text)
    current_blocks = extract_code_blocks(current_text)
    skill_name = entry.get("name", "Untitled Skill")

    session_id = entry.get("session_id")
    conversation_parts: list[str] = []

    if session_id is not None:
        history_sessions = read_all_history()
        session = next(
            (s for s in history_sessions if s.get("id") == session_id), None
        )
        if session and session.get("snapshot"):
            msgs = session["snapshot"].get("messages") or []
            if msgs:
                conversation_parts.append("[ORIGINAL CONVERSATION]")
                for m in msgs:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    if isinstance(content, list):
                        content = json.dumps(content)
                    conversation_parts.append(f"{role.upper()}: {content}")
                conversation_parts.append("")

    conversation_parts.append("[CURRENT SKILL]")
    conversation_parts.append(current_text)

    prompt = (
        "You are improving a crystallized skill. Below is the original "
        "conversation that produced this skill (if available), followed by "
        "the current skill content. Please regenerate an improved version "
        "with better rules and code.\n\n"
        + "\n".join(conversation_parts)
        + "\n\nOutput format: Rules section first, then fenced code blocks."
    )

    system_msg = (
        "You are a skill curator. Analyze the conversation and current skill. "
        "Produce clear architectural rules and well-structured code examples. "
        "Output rules as prose paragraphs, then code blocks in fenced format."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

    try:
        result = await run_in_threadpool(
            llm_chat, messages, model=req.model or DEFAULT_MODEL
        )
    except FreeLLMAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    new_rules = extract_rules(result.content)
    new_blocks = extract_code_blocks(result.content)

    if not new_rules and not new_blocks:
        new_rules = current_rules
        new_blocks = current_blocks

    md_content = build_markdown(skill_name, new_rules, new_blocks)
    tmp = skill_file.with_suffix(skill_file.suffix + ".tmp")
    tmp.write_text(md_content, encoding="utf-8")
    os.replace(tmp, skill_file)

    try:
        new_tags, new_category = _extract_tags_via_llm(
            llm_chat,
            skill_name,
            new_rules,
            len(new_blocks),
            model=req.model or DEFAULT_MODEL,
        )
    except Exception:
        new_tags = entry.get("tags") or []
        new_category = entry.get("category") or "Uncategorized"

    update_entry(filename, {
        "tags": new_tags,
        "category": new_category,
        "crystallized_at": datetime.now().isoformat(),
        "rule_chars": len(new_rules),
        "code_block_count": len(new_blocks),
    })

    updated = next((e for e in read_index() if e.get("filename") == filename), None)
    return {
        "entry": updated,
        "new_reply": result.content,
    }


# ---------------------------------------------------------------------------
# Tag extraction helper
# ---------------------------------------------------------------------------
def extract_tags_via_llm(
    skill_name: str, rules_text: str, code_block_count: int,
    *, model: str | None = None
) -> tuple[list[str], str]:
    """Call LLM to tag a skill. Returns (tags, category)."""
    from freellmapi_client import (
        DEFAULT_MODEL,
        chat as llm_chat,
    )
    return _extract_tags_via_llm(
        llm_chat, skill_name, rules_text, code_block_count,
        model=model,
    )


def _extract_tags_via_llm(
    llm_chat_fn,
    skill_name: str, rules_text: str, code_block_count: int,
    *, model: str | None = None
) -> tuple[list[str], str]:
    from freellmapi_client import DEFAULT_MODEL

    rules_snippet = rules_text[:1500] if rules_text else ""
    prompt = (
        "You are a library curator. Given this crystallized skill:\n\n"
        f"Name: {skill_name}\n"
        f"Rules: {rules_snippet}\n"
        f"Code blocks: {code_block_count} blocks\n\n"
        "Suggest:\n"
        '1. A short category (1-3 words, e.g. "Animation", "Layout", "UI Component", "Data")\n'
        "2. 2-5 keyword tags (lowercase, single words or short phrases like \"gsap\", \"flexbox\", \"dark-mode\")\n\n"
        "Respond with valid JSON only:\n"
        '{"category": "...", "tags": ["..."]}'
    )
    messages = [
        {"role": "system", "content": "You are a precise library curator. Respond with JSON only."},
        {"role": "user", "content": prompt},
    ]

    try:
        result = llm_chat_fn(messages, model=model or DEFAULT_MODEL)
        data = _parse_tag_json(result.content)
        return data.get("tags", []), data.get("category", "Uncategorized")
    except Exception:
        return [], "Uncategorized"


def _parse_tag_json(raw: str) -> dict[str, Any]:
    """Extract JSON from LLM response, with graceful fallback."""
    import re
    match = re.search(r'\{[^}]*\}', raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"category": "Uncategorized", "tags": []}
