"""Async OpenRouter client wrapper.

- Tenacity-based retry on transient errors (rate limits, timeouts, connection errors).
- Module-level semaphore caps total in-flight API calls regardless of how many call sites
  fan out concurrently. Configure via `set_global_concurrency()`.
- `chat_completion` returns the response text plus a provenance dict (usage, finish_reason,
  returned model, latency, prompt hash) so the runner can persist it for auditability.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time

import openai
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_APP_NAME,
    OPENROUTER_APP_URL,
)

_global_sem: asyncio.Semaphore | None = None
_global_sem_limit: int = 30


def set_global_concurrency(limit: int) -> None:
    """Set the global cap on in-flight API calls. Call before any chat_completion."""
    global _global_sem, _global_sem_limit
    _global_sem_limit = limit
    _global_sem = asyncio.Semaphore(limit)


def _ensure_sem() -> asyncio.Semaphore:
    global _global_sem
    if _global_sem is None:
        _global_sem = asyncio.Semaphore(_global_sem_limit)
    return _global_sem


def get_client() -> AsyncOpenAI:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": OPENROUTER_APP_URL,
            "X-Title": OPENROUTER_APP_NAME,
        },
    )


def prompt_hash(messages: list[dict]) -> str:
    """Stable hash of the messages payload, used to deduplicate identical calls in caches."""
    blob = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(
        (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)
    ),
    reraise=True,
)
async def chat_completion(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    response_format: dict | None = None,
) -> tuple[str, dict]:
    """Returns (text, provenance). Provenance is always populated; text may be empty."""
    sem = _ensure_sem()
    async with sem:
        t0 = time.monotonic()
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = await client.chat.completions.create(**kwargs)
        elapsed = time.monotonic() - t0

    choice = resp.choices[0]
    text = choice.message.content or ""
    usage = getattr(resp, "usage", None)
    provenance = {
        "model_requested": model,
        "model_returned": getattr(resp, "model", model),
        "finish_reason": getattr(choice, "finish_reason", None),
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        "latency_seconds": round(elapsed, 3),
        "prompt_hash": prompt_hash(messages),
    }
    return text, provenance
