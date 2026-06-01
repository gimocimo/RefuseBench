"""Tests for refusebench.runner helpers."""

from __future__ import annotations

import pytest

from refusebench.runner import response_hash


# ---------------------------------------------------------------------------
# response_hash — defensive guard against the empty-string collision (v0.4)
# ---------------------------------------------------------------------------

def test_response_hash_normal_text():
    """Non-empty text produces a stable 16-char hex hash."""
    h = response_hash("Hello, world!")
    assert isinstance(h, str)
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)
    # Determinism: same text -> same hash, so labels carry across runs.
    assert response_hash("Hello, world!") == h


def test_response_hash_distinct_for_distinct_text():
    assert response_hash("alpha") != response_hash("beta")


def test_response_hash_rejects_empty_string():
    """The v0.3 collision case: 4 cells with SHA-256("") = e3b0c44298fc1c14
    cross-matched in the calibration index. response_hash now refuses."""
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        response_hash("")


def test_response_hash_rejects_whitespace_only():
    """Whitespace-only is the same problem in disguise."""
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        response_hash("   \n\t  ")


def test_response_hash_rejects_none():
    """None is invalid input — should fail clearly, not silently coerce."""
    with pytest.raises((ValueError, AttributeError)):
        response_hash(None)  # type: ignore[arg-type]
