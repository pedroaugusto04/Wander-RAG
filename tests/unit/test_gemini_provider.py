"""Tests for GeminiProvider embedding retry helpers."""

from __future__ import annotations

from src.ai.llm.gemini_provider import GeminiProvider


class FakeError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_should_retry_on_429_status_code() -> None:
    exc = FakeError("quota exceeded", status_code=429)
    assert GeminiProvider._should_retry_embedding_error(exc)


def test_should_retry_on_resource_exhausted_message() -> None:
    exc = FakeError("RESOURCE_EXHAUSTED: retry later")
    assert GeminiProvider._should_retry_embedding_error(exc)


def test_should_not_retry_on_bad_request() -> None:
    exc = FakeError("invalid argument", status_code=400)
    assert not GeminiProvider._should_retry_embedding_error(exc)


def test_extract_retry_delay_seconds_from_message() -> None:
    exc = FakeError("Please retry in 6.001774627s.")
    delay = GeminiProvider._extract_retry_delay_seconds(exc)
    assert delay is not None
    assert delay > 6.0


def test_normalize_model_list_removes_empty_entries() -> None:
    result = GeminiProvider._normalize_model_list(["gemini-3.1-flash-lite", "", "  ", "gemma-3-12b"])
    assert result == ["gemini-3.1-flash-lite", "gemma-3-12b"]


def test_build_model_chain_deduplicates_preserving_order() -> None:
    chain = GeminiProvider._build_model_chain(
        "gemini-2.5-flash",
        ["gemini-3.1-flash-lite", "gemini-2.5-flash", "gemma-3-12b"],
    )
    assert chain == ["gemini-2.5-flash", "gemini-3.1-flash-lite", "gemma-3-12b"]
