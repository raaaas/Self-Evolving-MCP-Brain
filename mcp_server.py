"""
Self-Evolving MCP Brain — FastMCP server.

PHASE 1 scope:
  - One MCP server (FastMCP).
  - One MCP Resource, `get_rhythm_standards`, that reads every file inside the
    local `.mcp_skills/` directory and returns them as a single string.

PHASE 1 intentionally has NO tools, NO LLM calls, NO network. Those come later.
"""

from pathlib import Path

from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolve the skills directory relative to this file so the server works no
# matter what the current working directory is when it is launched.
PROJECT_ROOT = Path(__file__).resolve().parent
SKILLS_DIR = PROJECT_ROOT / ".mcp_skills"

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
mcp: FastMCP = FastMCP(
    name="self-evolving-brain",
    instructions=(
        "A self-evolving MCP brain. Its skill library lives in the local "
        ".mcp_skills/ directory and grows whenever a human approves a new "
        "pattern via the crystallization pipeline."
    ),
)


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------
@mcp.resource(
    "skills://rhythm-standards",
    name="get_rhythm_standards",
    description=(
        "Returns the entire current skill library as a single string. Each "
        "file in .mcp_skills/ is concatenated with a clear header so a "
        "consuming agent can read all crystallized rules and code patterns "
        "in one read."
    ),
    mime_type="text/plain",
)
def get_rhythm_standards() -> str:
    """Read every file in `.mcp_skills/` and return it as one string.

    The directory is read fresh on every call, so any file dropped into
    `.mcp_skills/` by the crystallization pipeline (Phase 3) is immediately
    visible to agents — no server restart required.

    Behaviour:
      * `.mcp_skills/` is created if it does not exist (defensive; the
        directory should already exist on disk).
      * Files are sorted by name for stable, deterministic output.
      * Each file is prefixed with a header line of the form
        `=== <filename> ===` and separated by a blank line.
      * An empty library returns an explicit, human-readable message instead
        of an empty string, so callers can distinguish "loaded nothing" from
        "failed to load".
    """
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)

    # Sorted for deterministic output across runs and filesystems.
    skill_files = sorted(
        p for p in SKILLS_DIR.iterdir() if p.is_file() and not p.name.startswith(".")
    )

    if not skill_files:
        return (
            "No crystallized skills yet. The .mcp_skills/ directory is empty. "
            "Once a pattern is analyzed and approved via the UI, it will appear here."
        )

    chunks: list[str] = []
    for path in skill_files:
        # Read as UTF-8; unknown bytes become replacement chars rather than
        # crashing the whole read if one bad file sneaks in.
        body = path.read_text(encoding="utf-8", errors="replace").rstrip()
        chunks.append(f"=== {path.name} ===\n{body}")

    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
# `transport` is intentionally left unset here. Run as an MCP server via:
#     fastmcp run mcp_server.py            # stdio (default, for agents)
#     python mcp_server.py                 # stdio via __main__ below
# Phase 4 will decide the concrete transport (host/port) when we wire it to
# the UI. Keeping it open avoids pinning a wrong transport prematurely.
if __name__ == "__main__":
    mcp.run()
