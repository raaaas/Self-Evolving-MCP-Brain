"""
File upload handler for the Self-Evolving MCP Brain.

Handles three upload types:
  1. ZIP archives — extract in memory, concatenate text-based files (HTML, JS,
     TS, Vue, CSS, etc.), skip node_modules/.git and binary files.
  2. Images (PNG, JPG, JPEG, GIF, WEBP) — encode to base64 for OpenAI vision.
  3. Single text/code files — read as-is.

Security:
  - ZIP path traversal: every extracted path is checked with .resolve() and
    must remain inside the virtual root. Symlinks are rejected.
  - Max total extracted size is capped (ZIP_BOMB_THRESHOLD).
  - Binary file detection via null-byte scan (not just extension).
"""

from __future__ import annotations

import base64
import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Directories to skip entirely during ZIP extraction (any depth).
SKIP_DIRS = frozenset({
    "node_modules", ".git", ".svn", ".hg", "__pycache__",
    ".tox", ".venv", "venv", ".mypy_cache", ".pytest_cache",
    "dist", "build", ".next", ".nuxt", "coverage",
})

# File extensions considered text-based and worth extracting from ZIPs.
TEXT_EXTENSIONS = frozenset({
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".mts", ".cts",
    ".vue", ".svelte",
    ".json", ".json5", ".yaml", ".yml", ".toml",
    ".py", ".rb", ".php", ".java", ".kt", ".rs", ".go", ".swift",
    ".c", ".cpp", ".h", ".hpp", ".cs",
    ".sh", ".bash", ".zsh", ".fish",
    ".md", ".mdx", ".txt", ".csv", ".xml", ".svg",
    ".env", ".env.local", ".env.development",
    ".astro", ".md",
    ".wasm",  # text sometimes, skip if binary
})

# Extensions for image files (base64 → vision).
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})

# Safety: max total bytes extracted from a ZIP (default 50 MB).
ZIP_BOMB_THRESHOLD = 50 * 1024 * 1024

# Max single-image size for base64 encoding (10 MB).
MAX_IMAGE_BYTES = 10 * 1024 * 1024


@dataclass
class UploadResult:
    """Normalized result of a file upload, ready to pass to the LLM."""

    upload_type: Literal["zip_text", "image", "text_file", "unknown"]
    content: Any  # str for text types, list[dict] for image vision blocks
    summary: str  # human-readable summary for logging/UI
    filename: str  # original uploaded filename


# ---------------------------------------------------------------------------
# Binary detection
# ---------------------------------------------------------------------------
def _is_binary(data: bytes) -> bool:
    """Heuristic: if the first 8192 bytes contain a NULL byte, treat as binary."""
    return b"\x00" in data[:8192]


# ---------------------------------------------------------------------------
# ZIP handling (secure in-memory extraction)
# ---------------------------------------------------------------------------
def _safe_zip_path(zip_info: zipfile.ZipInfo, virtual_root: Path) -> Path | None:
    """Validate an extracted path is safe (no traversal, no symlink, not skipped).

    Returns the resolved target Path if safe, or None if it should be skipped.
    """
    # Skip directories.
    if zip_info.is_dir():
        return None

    name = zip_info.filename

    # Normalize to POSIX and check for traversal.
    normalized = PurePosixPath(name)
    parts = normalized.parts

    # Skip if any component is in SKIP_DIRS.
    for part in parts:
        if part in SKIP_DIRS:
            return None

    # Reject absolute paths or traversal (..).
    if normalized.is_absolute() or ".." in parts:
        return None

    # Resolve against virtual root to catch any trickery.
    target = (virtual_root / str(normalized)).resolve()
    try:
        target.relative_to(virtual_root.resolve())
    except ValueError:
        return None  # escaped the root

    return target


def process_zip(file_bytes: bytes, original_filename: str) -> UploadResult:
    """Extract a ZIP in memory, concatenate readable text files.

    Raises ValueError on invalid ZIP or ZIP bomb.
    """
    virtual_root = Path("/__virtual_zip_root__")
    chunks: list[str] = []
    total_extracted = 0
    file_count = 0
    skipped_binary = 0
    skipped_unrecognized = 0

    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            # Sort for deterministic output.
            for info in sorted(zf.infolist(), key=lambda z: z.filename):
                safe = _safe_zip_path(info, virtual_root)
                if safe is None:
                    continue

                ext = Path(info.filename).suffix.lower()

                # Only extract recognized text extensions.
                if ext not in TEXT_EXTENSIONS:
                    skipped_unrecognized += 1
                    continue

                data = zf.read(info.filename)
                total_extracted += len(data)

                if total_extracted > ZIP_BOMB_THRESHOLD:
                    raise ValueError(
                        f"ZIP extraction exceeded {ZIP_BOMB_THRESHOLD} bytes "
                        f"(potential zip bomb). Aborted after {file_count} files."
                    )

                # Skip if binary despite text extension.
                if _is_binary(data):
                    skipped_binary += 1
                    continue

                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    skipped_binary += 1
                    continue

                header_name = str(PurePosixPath(info.filename))
                chunks.append(f"--- {header_name} ---\n{text}")
                file_count += 1

    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid ZIP file: {exc}") from exc

    if not chunks:
        return UploadResult(
            upload_type="zip_text",
            content="",
            summary=(
                f"ZIP '{original_filename}' contained no readable code files. "
                f"(skipped {skipped_unrecognized} unrecognized, "
                f"{skipped_binary} binary)"
            ),
            filename=original_filename,
        )

    combined = "\n\n".join(chunks)
    return UploadResult(
        upload_type="zip_text",
        content=combined,
        summary=(
            f"ZIP '{original_filename}': extracted {file_count} code files "
            f"({len(combined)} chars). "
            f"Skipped {skipped_unrecognized} unrecognized, {skipped_binary} binary."
        ),
        filename=original_filename,
    )


# ---------------------------------------------------------------------------
# Image handling (base64 for vision)
# ---------------------------------------------------------------------------
def process_image(file_bytes: bytes, original_filename: str) -> UploadResult:
    """Encode an image to base64 and return an OpenAI vision content block."""
    if len(file_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image too large: {len(file_bytes)} bytes "
            f"(max {MAX_IMAGE_BYTES}). Compress or resize before uploading."
        )

    ext = Path(original_filename).suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime = mime_map.get(ext, "image/png")
    b64 = base64.standard_b64encode(file_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    vision_block: list[dict[str, Any]] = [
        {"type": "text", "text": f"Analyze this image (screenshot) named '{original_filename}'."},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    return UploadResult(
        upload_type="image",
        content=vision_block,
        summary=f"Image '{original_filename}' ({len(file_bytes)} bytes, {mime}) encoded as base64.",
        filename=original_filename,
    )


# ---------------------------------------------------------------------------
# Single text file handling
# ---------------------------------------------------------------------------
def process_text_file(file_bytes: bytes, original_filename: str) -> UploadResult:
    """Read a single text/code file."""
    if _is_binary(file_bytes):
        raise ValueError(
            f"File '{original_filename}' appears to be binary. "
            "Only text/code files are supported for direct upload."
        )
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"File '{original_filename}' is not valid UTF-8 text."
        ) from exc

    return UploadResult(
        upload_type="text_file",
        content=text,
        summary=f"Text file '{original_filename}' ({len(text)} chars).",
        filename=original_filename,
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def process_upload(file_bytes: bytes, filename: str) -> UploadResult:
    """Route to the correct processor based on extension."""
    ext = Path(filename).suffix.lower()

    if ext == ".zip":
        return process_zip(file_bytes, filename)
    if ext in IMAGE_EXTENSIONS:
        return process_image(file_bytes, filename)
    if ext in TEXT_EXTENSIONS or ext in {
        ".txt", ".log", ".cfg", ".ini", ".conf", ".dockerfile",
        ".makefile", ".cmake", ".gradle",
    }:
        return process_text_file(file_bytes, filename)

    # Unknown extension — try as text first, fall back to rejection.
    if not _is_binary(file_bytes):
        try:
            text = file_bytes.decode("utf-8")
            return UploadResult(
                upload_type="text_file",
                content=text,
                summary=f"File '{filename}' ({len(text)} chars, extension '{ext}' unrecognized, treated as text).",
                filename=filename,
            )
        except UnicodeDecodeError:
            pass

    raise ValueError(
        f"Unsupported file type: '{filename}' (extension '{ext}'). "
        "Upload .zip, .png/.jpg/.jpeg, or text/code files."
    )
