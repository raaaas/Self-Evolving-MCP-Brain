"""
Crystallization pipeline.

Takes the LLM's final accepted response (a single string that may contain a
rules section and fenced code blocks) and persists it as a structured Markdown
file inside .mcp_skills/, where the FastMCP `skills://rhythm-standards`
resource reads it fresh on every call — so newly crystallized skills are
instantly available to agents with no server restart.

The module is deliberately HTTP-free so it can be unit-tested in isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SKILLS_DIR = PROJECT_ROOT / ".mcp_skills"

# Single source of truth for the fenced-code regex: this MUST match the
# frontend's extractCode() in frontend/index.html so what the user previews is
# exactly what gets crystallized.
FENCE_RE = re.compile(r"```(\w+)?\n([\s\S]*?)```", re.MULTILINE)

# What counts as a "rule" heading inside the LLM reply. We look for the most
# common phrasings; if none match, we treat all non-code text as rules.
RULES_HEADING_RE = re.compile(
    r"(?im)^\s*(?:#+\s*)?(?:core\s+(?:architectural\s+)?rules?|architectural\s+patterns?|"
    r"rules?\s+and\s+standards?|coding\s+standards?|ui/?ux\s+rules?|summary|analysis)\b[^\n]*$"
)


@dataclass
class CrystallizeResult:
    """Outcome of a crystallize operation."""

    saved_path: Path
    skill_name: str
    filename: str
    rules_text: str
    code_blocks: list[tuple[str, str]]  # [(lang, code), ...]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Return [(lang, code), ...] for every fenced block. lang is '' if unset."""
    return [
        ((m.group(1) or "").lower(), m.group(2).rstrip("\n"))
        for m in FENCE_RE.finditer(text)
    ]


def extract_rules(text: str) -> str:
    """Pull the rules portion out of the LLM reply.

    Strategy (no guessing about reply structure):
      1. If a recognizable rules heading exists, take the text from that
         heading up to the next heading or the first code fence.
      2. Otherwise, take all non-code prose (everything outside fences).
    """
    if not text.strip():
        return ""

    m = RULES_HEADING_RE.search(text)
    if m:
        # Take text AFTER the heading line, up to the next heading or first fence.
        # (build_markdown supplies the canonical "## Core Rules" heading, so we
        # must not include the heading line itself or we'd duplicate it.)
        rest = text[m.end():]
        next_heading = re.search(r"(?m)^\s*#+\s+\S", rest)
        first_fence = rest.find("```")
        cutoffs = [x for x in (next_heading.start() if next_heading else None,
                               first_fence if first_fence != -1 else None)
                   if x is not None]
        end = min(cutoffs) if cutoffs else len(rest)
        rules = rest[:end].strip()
        return rules

    # Fallback: all prose outside code fences.
    stripped = FENCE_RE.sub("", text).strip()
    return stripped


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------
def _slugify(name: str, *, max_len: int = 40) -> str:
    """Make a filesystem-safe slug. 'GSAP Nav Animations!' -> 'gsap_nav_animations'."""
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] or "skill"


def make_filename(skill_name: str, *, now: datetime | None = None) -> str:
    """skill_<slug>_<YYYYMMDDHHMM>.md  — sanitized name + timestamp for uniqueness."""
    when = (now or datetime.now()).strftime("%Y%m%d%H%M")
    return f"skill_{_slugify(skill_name)}_{when}.md"


def make_skill_name_default(text: str) -> str:
    """Guess a default skill name from the reply's first heading or first line."""
    # Prefer the first markdown heading.
    m = re.search(r"(?m)^\s*#+\s*(.+?)\s*$", text)
    if m:
        return m.group(1)
    # Else first non-empty line, truncated.
    for line in text.splitlines():
        line = line.strip(" -*\t")
        if line:
            return line[:60]
    return "untitled-skill"


# ---------------------------------------------------------------------------
# Markdown assembly
# ---------------------------------------------------------------------------
def build_markdown(skill_name: str, rules_text: str,
                   code_blocks: list[tuple[str, str]]) -> str:
    """Assemble the exact Markdown shape the human specified:

        # [Skill Name]

        ## Core Rules
        [Rules]

        ## Architecture Boilerplate
        [Code Snippets]

    Rules and code are pulled from the LLM reply; nothing is invented.
    If either is empty, we write an explicit placeholder rather than fake content.
    """
    rules_section = rules_text.strip() if rules_text.strip() else (
        "_(No explicit rules were extracted from the approved response.)_"
    )

    if code_blocks:
        boilerplate_parts = []
        for i, (lang, code) in enumerate(code_blocks, start=1):
            label = lang.upper() if lang else "CODE"
            boilerplate_parts.append(f"### Snippet {i} — {label}\n\n```{lang}\n{code}\n```")
        boilerplate = "\n\n".join(boilerplate_parts)
    else:
        boilerplate = "_(No code snippets were present in the approved response.)_"

    # Stable footer so the provenance of every skill is auditable.
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    footer = (
        f"\n\n---\n_Crystallized on {timestamp} via the Self-Evolving MCP Brain._"
    )

    return (
        f"# {skill_name.strip()}\n\n"
        f"## Core Rules\n\n{rules_section}\n\n"
        f"## Architecture Boilerplate\n\n{boilerplate}"
        f"{footer}\n"
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_skill(skill_name: str, rules_text: str,
               code_blocks: list[tuple[str, str]],
               *, now: datetime | None = None) -> CrystallizeResult:
    """Build + write the Markdown file to .mcp_skills/. Returns the result.

    The file is written atomically (write to temp, rename) so a partial write
    can never appear in the skills directory mid-way.
    """
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    filename = make_filename(skill_name, now=now)
    target = SKILLS_DIR / filename

    # Guard against accidental clobber from identical slug+timestamp collisions.
    i = 1
    while target.exists():
        stem = target.stem
        target = SKILLS_DIR / f"{stem}_{i}.md"
        i += 1

    markdown = build_markdown(skill_name, rules_text, code_blocks)

    # Atomic write: temp file in same dir, then os.replace.
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(markdown, encoding="utf-8")
    tmp.replace(target)

    return CrystallizeResult(
        saved_path=target,
        skill_name=skill_name,
        filename=target.name,
        rules_text=rules_text,
        code_blocks=code_blocks,
    )


def crystallize_reply(reply: str, skill_name: str | None = None,
                      *, now: datetime | None = None) -> CrystallizeResult:
    """One-shot: extract from a raw LLM reply, then save.

    Convenience wrapper used by the /crystallize endpoint.
    """
    rules = extract_rules(reply)
    blocks = extract_code_blocks(reply)
    name = (skill_name or "").strip() or make_skill_name_default(reply)
    return save_skill(name, rules, blocks, now=now)
