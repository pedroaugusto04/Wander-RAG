"""Tests for GeminiProvider embedding retry helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

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


@pytest.mark.asyncio
async def test_generate_embeddings_retries_same_model_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeModels:
        def __init__(self) -> None:
            self.attempt = 0

        async def embed_content(self, *, model: str, contents: list[Any], config: Any) -> Any:
            calls.append({"model": model, "config": config})
            self.attempt += 1
            if self.attempt < 3:
                raise FakeError("quota exceeded", status_code=429)
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3]) for _ in contents]
            )

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.aio = SimpleNamespace(models=FakeModels())

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("src.ai.llm.gemini_provider.genai.Client", FakeClient)
    monkeypatch.setattr("src.ai.llm.gemini_provider.asyncio.sleep", no_sleep)

    provider = GeminiProvider(
        api_key="test-key",
        embedding_model="models/gemini-embedding-2-preview",
        embedding_max_retries=3,
        embedding_requests_per_minute=0,
    )

    embeddings = await provider.generate_embeddings(["teste"])

    assert embeddings == [[0.1, 0.2, 0.3]]
    assert calls == [
        {"model": "models/gemini-embedding-2-preview", "config": None},
        {"model": "models/gemini-embedding-2-preview", "config": None},
        {"model": "models/gemini-embedding-2-preview", "config": None},
    ]


@pytest.mark.asyncio
async def test_generate_embeddings_stops_after_max_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeModels:
        async def embed_content(self, *, model: str, contents: list[Any], config: Any) -> Any:
            if not contents:
                raise AssertionError("Expected at least one content item")
            calls.append({"model": model, "config": config})
            raise FakeError("quota exceeded", status_code=429)

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.aio = SimpleNamespace(models=FakeModels())

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("src.ai.llm.gemini_provider.genai.Client", FakeClient)
    monkeypatch.setattr("src.ai.llm.gemini_provider.asyncio.sleep", no_sleep)

    provider = GeminiProvider(
        api_key="test-key",
        embedding_model="models/gemini-embedding-2-preview",
        embedding_max_retries=3,
        embedding_requests_per_minute=0,
    )

    with pytest.raises(RuntimeError, match="Embedding failed on 'models/gemini-embedding-2-preview'"):
        await provider.generate_embeddings(["teste"])

    assert calls == [
        {"model": "models/gemini-embedding-2-preview", "config": None},
        {"model": "models/gemini-embedding-2-preview", "config": None},
        {"model": "models/gemini-embedding-2-preview", "config": None},
    ]


@pytest.mark.asyncio
async def test_generate_embeddings_forwards_dimensions_and_task_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_config: Any = None

    class FakeModels:
        async def embed_content(self, *, model: str, contents: list[Any], config: Any) -> Any:
            nonlocal seen_config
            assert model == "models/gemini-embedding-2-preview"
            assert len(contents) == 2
            seen_config = config
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3]) for _ in contents]
            )

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.aio = SimpleNamespace(models=FakeModels())

    monkeypatch.setattr("src.ai.llm.gemini_provider.genai.Client", FakeClient)

    provider = GeminiProvider(
        api_key="test-key",
        embedding_model="models/gemini-embedding-2-preview",
        embedding_requests_per_minute=0,
    )

    embeddings = await provider.generate_embeddings(
        ["doc 1", "doc 2"],
        dimensions=768,
        task_type="RETRIEVAL_DOCUMENT",
    )

    assert embeddings == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert seen_config == {
        "output_dimensionality": 768,
        "task_type": "RETRIEVAL_DOCUMENT",
    }
