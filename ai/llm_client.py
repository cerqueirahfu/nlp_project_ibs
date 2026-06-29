"""Thin wrapper around the OpenAI API for the PricePulse AI features.

Used by the Streamlit dashboard for (1) on-demand insight summaries,
(2) the product review Pros/Cons summary, and (3) the dataset chat page.
All calls go through `complete()`.

Design goals:
- **No keys in code.** The OpenAI SDK reads the `OPENAI_API_KEY` environment
  variable. Set it locally (`$env:OPENAI_API_KEY = "sk-..."`) or on EC2.
- **Degrade gracefully.** If the `openai` package is missing or no API key is
  set, `available()` returns False and the UI hides/disables the AI bits
  instead of crashing. `complete()` raises `LLMError` with a readable message
  that the UI surfaces.

Config via environment variables (all optional except the key):
    OPENAI_API_KEY      required for AI features to work
    OPENAI_MODEL        default: gpt-4o-mini  (cheap + capable; ~$0.15/$0.60 per Mtok)
    OPENAI_BASE_URL     optional; for Azure/OpenAI-compatible gateways
    OPENAI_TIMEOUT      per-request timeout in seconds (default: 30) so a slow
                        response can't hang the Streamlit spinner forever
    OPENAI_MAX_RETRIES  retries on transient errors (rate limit / connection),
                        with the SDK's exponential backoff (default: 3)
"""
from __future__ import annotations

import os
from functools import lru_cache

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 30.0      # seconds; bounds how long a call can block the UI
DEFAULT_MAX_RETRIES = 3     # SDK retries transient errors with exponential backoff


class LLMError(RuntimeError):
    """Raised when an AI call fails for a reason worth showing the user."""


def model_id() -> str:
    return os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)


def _timeout() -> float:
    try:
        return float(os.environ.get("OPENAI_TIMEOUT", DEFAULT_TIMEOUT))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT


def _max_retries() -> int:
    try:
        return int(os.environ.get("OPENAI_MAX_RETRIES", DEFAULT_MAX_RETRIES))
    except (TypeError, ValueError):
        return DEFAULT_MAX_RETRIES


@lru_cache(maxsize=1)
def _client():
    """Build a cached OpenAI client, or raise LLMError.

    The SDK resolves the API key from OPENAI_API_KEY automatically.
    """
    try:
        from openai import OpenAI  # imported lazily so the app runs without openai
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise LLMError(
            "The `openai` package is not installed. Run `pip install openai` "
            "to enable AI features."
        ) from exc
    kwargs = {"timeout": _timeout(), "max_retries": _max_retries()}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


@lru_cache(maxsize=1)
def available() -> bool:
    """True if AI calls are likely to work (openai importable + API key set).

    Cheap check — no network call. A True result can still fail later (bad key,
    rate limit), which `complete()` reports cleanly.
    """
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("OPENAI_API_KEY"))


def complete(
    messages: list[dict],
    *,
    system: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """Send a chat-style request to OpenAI and return the text.

    `messages` is the standard chat format: [{"role": "user"|"assistant",
    "content": "..."}] — already OpenAI-compatible. `max_tokens` is always
    capped to keep cost bounded.
    """
    try:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            AuthenticationError,
            NotFoundError,
            OpenAIError,
            RateLimitError,
        )
    except ImportError as exc:  # pragma: no cover
        raise LLMError("The `openai` package is not installed.") from exc

    full_messages = ([{"role": "system", "content": system}] if system else []) + messages

    client = _client()
    try:
        resp = client.chat.completions.create(
            model=model_id(),
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
    except AuthenticationError as exc:
        raise LLMError(
            "OpenAI rejected the API key. Check that OPENAI_API_KEY is set and valid."
        ) from exc
    except RateLimitError as exc:
        raise LLMError(
            "OpenAI rate limit or insufficient credit. Check your account usage/billing."
        ) from exc
    except APITimeoutError as exc:  # subclass of APIConnectionError — must come first
        raise LLMError(
            f"OpenAI timed out after {_timeout():.0f}s. Try again, narrow the "
            "selection, or raise OPENAI_TIMEOUT."
        ) from exc
    except NotFoundError as exc:
        raise LLMError(
            f"Model `{model_id()}` not found for this account. "
            "Set OPENAI_MODEL to a model you have access to (e.g. gpt-4o-mini)."
        ) from exc
    except APIConnectionError as exc:
        raise LLMError(f"Could not reach OpenAI: {exc}") from exc
    except OpenAIError as exc:
        raise LLMError(f"OpenAI call failed: {exc}") from exc
    except (KeyError, IndexError, AttributeError) as exc:
        raise LLMError("Unexpected response shape from OpenAI.") from exc
