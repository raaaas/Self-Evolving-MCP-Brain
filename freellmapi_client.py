"""
Thin raw-HTTP client for the local freellmapi proxy.

Why this file is separate from ui_server.py:
  - Keeps the OpenAI-wire-format details in one auditable place.
  - Easy to unit-test the payload assembly without spinning up FastAPI.

freellmapi is NOT a Python package; it is a Node/Docker proxy that exposes an
OpenAI-compatible `/v1` endpoint. Per project decision we use raw `requests`
(no `openai` SDK) to keep the dependency surface tiny.

Authoritative behaviour (from the freellmapi README):
  - POST {base}/v1/chat/completions with {model, messages:[{role,content}]}
  - Auth: Bearer token, value from the operator's Keys page. Read from env.
  - model "auto" lets the proxy's router choose the best model.
  - Response is standard OpenAI: choices[0].message.content
  - GET  {base}/v1/models lists available models.

No guessing: every constant below is anchored to the README.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
import requests

load_dotenv()

# ---------------------------------------------------------------------------
# Config (no hardcoded secrets — read from the environment)
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = os.getenv("FREELLMAPI_BASE_URL", "http://localhost:3001")
DEFAULT_MODEL = os.getenv("FREELLMAPI_MODEL", "auto")
HTTP_TIMEOUT = float(os.getenv("FREELLMAPI_TIMEOUT", "60"))

# Retry: attempts, backoff, and which status codes trigger a retry.
MAX_RETRIES = int(os.getenv("FREELLMAPI_MAX_RETRIES", "3"))
RETRY_BACKOFF = float(os.getenv("FREELLMAPI_RETRY_BACKOFF", "1.5"))
RETRY_STATUSES = (429, 500, 502, 503, 504)


class FreeLLMAPIError(RuntimeError):
    """Raised when the proxy is unreachable or returns a non-2xx response."""


class FreeLLMAPITimeout(FreeLLMAPIError):
    """Raised when a raw HTTP call to the proxy exceeds HTTP_TIMEOUT after retries."""


@dataclass
class ChatResult:
    """Normalised result of one chat-completions call."""

    content: str
    model: str  # the model the proxy reports it actually used (may differ from input)
    routed_via: str | None  # X-Routed-Via header if present


def _bearer_token() -> str:
    """Read the proxy's unified key from the environment.

    Returns the raw token value (the caller adds the 'Bearer ' prefix).
    Raises a clear, actionable error if it is missing.
    """
    token = os.getenv("FREELLMAPI_KEY")
    if not token:
        raise FreeLLMAPIError(
            "FREELLMAPI_KEY is not set. Create a key on the freellmapi proxy's "
            "Keys page and export it (or put it in .env)."
        )
    return token


def _auth_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_bearer_token()}",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------
def _retry_request(method: str, url: str, **kwargs) -> requests.Response:
    """Execute an HTTP request with a single deadline and retry on transient errors.

    Deliberately does NOT retry on timeout. Retrying a timed-out request
    through a slow proxy creates a death spiral: the old request is still
    running inside the proxy, and the new request piles on additional load,
    making the proxy even slower. Use a single generous timeout instead.

    Retries on transient server errors (429, 5xx) with exponential backoff,
    up to MAX_RETRIES, bounded by a total wall-clock deadline so retries
    cannot extend the wait indefinitely.
    """
    start = time.monotonic()
    last_exc: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        # How much time remains for this attempt?
        elapsed = time.monotonic() - start
        remaining = HTTP_TIMEOUT - elapsed
        if remaining <= 0:
            raise FreeLLMAPITimeout(
                f"freellmapi {method} {url} total deadline of {HTTP_TIMEOUT}s "
                f"exhausted after {attempt} attempt(s)"
            )

        # Clamp per-attempt timeout to the remaining deadline.
        attempt_timeout = min(remaining, HTTP_TIMEOUT)
        try:
            resp = requests.request(method, url, timeout=attempt_timeout, **{
                k: v for k, v in kwargs.items() if k != "timeout"
            })
        except requests.Timeout:
            # DO NOT retry on timeout — the proxy is slow, adding more
            # requests only makes it slower.
            raise FreeLLMAPITimeout(
                f"freellmapi {method} {url} timed out after {attempt_timeout:.0f}s"
            )
        except requests.RequestException as exc:
            raise FreeLLMAPIError(
                f"Cannot reach freellmapi: {exc}"
            ) from exc

        if resp.status_code in RETRY_STATUSES and attempt < MAX_RETRIES:
            delay = RETRY_BACKOFF ** attempt
            # Don't sleep past the total deadline.
            if time.monotonic() - start + delay < HTTP_TIMEOUT:
                time.sleep(delay)
            continue

        return resp

    assert last_exc is not None
    raise last_exc  # type: ignore[unreachable]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def list_models(base_url: str = DEFAULT_BASE_URL) -> list[str]:
    """GET /v1/models — return list of model ids. Raises FreeLLMAPIError on failure."""
    resp = _retry_request(
        "GET",
        f"{base_url.rstrip('/')}/v1/models",
        headers=_auth_headers(),
    )

    if resp.status_code != 200:
        raise FreeLLMAPIError(
            f"GET /v1/models failed: HTTP {resp.status_code} — {resp.text[:300]}"
        )
    data = resp.json()
    # OpenAI shape: {"data":[{"id":...,"object":"model"}, ...]}
    return [m.get("id", "?") for m in data.get("data", [])]


def chat(
    messages: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.3,
    max_tokens: int | None = None,
) -> ChatResult:
    """Send a chat-completions request to the freellmapi proxy.

    `messages` is OpenAI-shaped: [{"role": "system|user|assistant", "content": ...}].
    Content may be a plain string OR an OpenAI vision-style list of content
    blocks (e.g. [{"type":"text","text":...},{"type":"image_url","image_url":{"url":...}}]).
    We intentionally do not validate content shape — the proxy is the authority.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    resp = _retry_request(
        "POST",
        f"{base_url.rstrip('/')}/v1/chat/completions",
        headers=_auth_headers(),
        json=payload,
    )

    if resp.status_code >= 400:
        raise FreeLLMAPIError(
            f"freellmapi HTTP {resp.status_code}: {resp.text[:500]}"
        )

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise FreeLLMAPIError(
            f"Unexpected freellmapi response shape: {str(data)[:500]}"
        ) from exc

    return ChatResult(
        content=content or "",
        model=data.get("model", model),
        routed_via=resp.headers.get("X-Routed-Via"),
    )
