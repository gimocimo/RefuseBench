"""Tests for refusebench.models.chat_completion — empty-response handling.

The v0.3 inspection-run audit (Codex review #3) surfaced 4 empty responses
that slipped past the original EmptyResponseError check because the SDK
returned a valid choice/message with empty .content. These tests pin the
v0.4 fix: ALL forms of empty completion — no choices, no message, content
None, content whitespace-only — must raise EmptyResponseError so tenacity
can retry and (if persistent) the cell lands in failures.json for resume.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import tenacity

from refusebench.models import EmptyResponseError, chat_completion


@pytest.fixture(autouse=True)
def _instant_retries(monkeypatch):
    """Make tenacity retries fire instantly so tests don't wait the exponential
    backoff. monkeypatch restores after each test."""
    monkeypatch.setattr(chat_completion.retry, "wait", tenacity.wait_none())


@pytest.fixture
def mock_client():
    """An AsyncOpenAI-shaped mock with a configurable .chat.completions.create()."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


def make_response(content, finish_reason="stop", completion_tokens=10):
    """Build a ChatCompletion-shaped mock with the given content."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason
    resp = MagicMock()
    resp.choices = [choice]
    resp.model = "test-model"
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=completion_tokens, total_tokens=20)
    resp.error = None
    return resp


def make_no_choices_response(error_message="upstream provider failure"):
    """Build a response with choices=None — the provider-error-disguised-as-200 case."""
    resp = MagicMock()
    resp.choices = None
    resp.model = "test-model"
    resp.usage = None
    resp.error = {"message": error_message}
    return resp


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

async def test_normal_response_returns_text_and_provenance(mock_client):
    mock_client.chat.completions.create.return_value = make_response("Hello, world!")
    text, prov = await chat_completion(
        mock_client, "test-model", [{"role": "user", "content": "hi"}]
    )
    assert text == "Hello, world!"
    assert prov["model_returned"] == "test-model"
    assert prov["finish_reason"] == "stop"
    assert prov["completion_tokens"] == 10
    assert "prompt_hash" in prov


# ---------------------------------------------------------------------------
# Empty-response paths — the v0.4 fix
# ---------------------------------------------------------------------------

async def test_choices_none_raises_no_completion_choice(mock_client):
    """The original v0.3 case: OpenRouter returns 200 with no choices."""
    mock_client.chat.completions.create.return_value = make_no_choices_response(
        "provider down"
    )
    with pytest.raises(EmptyResponseError, match="no completion choice"):
        await chat_completion(
            mock_client, "test-model", [{"role": "user", "content": "hi"}]
        )


async def test_empty_string_content_raises_empty_content(mock_client):
    """content="" — the case that slipped past v0.3 and produced 4 empty responses."""
    mock_client.chat.completions.create.return_value = make_response("", finish_reason="stop")
    with pytest.raises(EmptyResponseError, match="empty content"):
        await chat_completion(
            mock_client, "test-model", [{"role": "user", "content": "hi"}]
        )


async def test_none_content_raises_empty_content(mock_client):
    """content=None — what reasoning models return when they burn max_tokens on reasoning."""
    mock_client.chat.completions.create.return_value = make_response(
        None, finish_reason="length", completion_tokens=4096
    )
    with pytest.raises(EmptyResponseError, match="empty content"):
        await chat_completion(
            mock_client, "test-model", [{"role": "user", "content": "hi"}]
        )


async def test_whitespace_only_content_raises_empty_content(mock_client):
    """Whitespace-only counts as empty — there's nothing to evaluate."""
    mock_client.chat.completions.create.return_value = make_response("   \n\t  ")
    with pytest.raises(EmptyResponseError, match="empty content"):
        await chat_completion(
            mock_client, "test-model", [{"role": "user", "content": "hi"}]
        )


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------

async def test_empty_response_triggers_full_retry_loop(mock_client):
    """Persistent empties should exhaust 4 tenacity attempts before re-raising."""
    mock_client.chat.completions.create.return_value = make_response("")
    with pytest.raises(EmptyResponseError):
        await chat_completion(
            mock_client, "test-model", [{"role": "user", "content": "hi"}]
        )
    assert mock_client.chat.completions.create.call_count == 4


async def test_transient_empty_response_recovers_on_retry(mock_client):
    """First call empty -> retry -> second call returns content -> success."""
    mock_client.chat.completions.create.side_effect = [
        make_response(""),                # first call: empty, triggers retry
        make_response("recovered!"),      # second call: success
    ]
    text, prov = await chat_completion(
        mock_client, "test-model", [{"role": "user", "content": "hi"}]
    )
    assert text == "recovered!"
    assert mock_client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# Error-message quality
# ---------------------------------------------------------------------------

async def test_error_includes_finish_reason_and_tokens(mock_client):
    """The error string must surface finish_reason + completion_tokens — these are
    what diagnose 'thinking model burned its budget' vs 'provider returned a blip'."""
    mock_client.chat.completions.create.return_value = make_response(
        None, finish_reason="length", completion_tokens=4096
    )
    with pytest.raises(EmptyResponseError) as exc_info:
        await chat_completion(
            mock_client, "test-model", [{"role": "user", "content": "hi"}]
        )
    msg = str(exc_info.value)
    assert "finish_reason=length" in msg
    assert "completion_tokens=4096" in msg
    assert "test-model" in msg
