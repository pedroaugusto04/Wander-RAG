"""Google Gemini LLM provider implementation using the google-genai SDK."""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from typing import TYPE_CHECKING

from google import genai
from google.genai import types

from src.ai.llm.base import EmbeddingTaskType, LLMProvider, LLMResponse
from src.config.settings import (
    DEFAULT_EMBEDDING_BASE_RETRY_SECONDS,
    DEFAULT_EMBEDDING_MAX_RETRIES,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REQUESTS_PER_MINUTE,
    DEFAULT_LLM_MODEL,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence


class GeminiProvider(LLMProvider):
    """Google Gemini provider using the official google-genai SDK."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_LLM_MODEL,
        fallback_models: Sequence[str] | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_requests_per_minute: int = DEFAULT_EMBEDDING_REQUESTS_PER_MINUTE,
        embedding_max_retries: int = DEFAULT_EMBEDDING_MAX_RETRIES,
        embedding_base_retry_seconds: float = DEFAULT_EMBEDDING_BASE_RETRY_SECONDS,
    ) -> None:
        self.model = model
        self.fallback_models = self._normalize_model_list(fallback_models)
        self._generate_models = self._build_model_chain(self.model, self.fallback_models)

        self.embedding_model = embedding_model
        self._cached_embedding_dimension: int | None = None
        self.embedding_max_retries = max(1, embedding_max_retries)
        self.embedding_base_retry_seconds = max(0.1, embedding_base_retry_seconds)
        self._embed_min_interval_seconds = (
            60.0 / embedding_requests_per_minute if embedding_requests_per_minute > 0 else 0.0
        )
        self._embed_lock = asyncio.Lock()
        self._last_embed_request_at = 0.0

        self.client = genai.Client(api_key=api_key)

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Call Gemini to generate a chat completion."""

        contents: list[types.Content] = []
        system_instruction: str | None = None

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction = content
                continue

            gemini_role = "model" if role == "assistant" else "user"

            contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=content)],
                )
            )

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction,
        )

        response = None
        active_model = self.model
        last_exc: Exception | None = None

        for index, candidate_model in enumerate(self._generate_models):
            active_model = candidate_model
            try:
                response = await self.client.aio.models.generate_content(
                    model=candidate_model,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:
                last_exc = exc
                is_last_model = index >= len(self._generate_models) - 1
                if not self._should_retry_embedding_error(exc) or is_last_model:
                    raise RuntimeError(
                        f"Text generation failed on '{candidate_model}': "
                        f"{self._format_exception_summary(exc)}"
                    ) from None

                logger.warning(
                    "Text generation failed on model '%s'. Trying fallback model: %s",
                    candidate_model,
                    type(exc).__name__,
                )
                continue

            break

        if response is None:
            if last_exc is not None:
                raise RuntimeError(
                    f"Text generation failed on all configured models: "
                    f"{self._format_exception_summary(last_exc)}"
                ) from None
            raise RuntimeError("No response returned by any configured generation model")

        usage: dict[str, int] = {}
        if response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                "completion_tokens": response.usage_metadata.candidates_token_count or 0,
                "total_tokens": response.usage_metadata.total_token_count or 0,
            }

        text = response.text or ""
        finish_reason = "stop"

        if response.candidates and response.candidates[0].finish_reason:
            finish_reason = response.candidates[0].finish_reason.name

        logger.debug(
            "Gemini response: model=%s, tokens=%s, finish=%s",
            active_model,
            usage.get("total_tokens", "?"),
            finish_reason,
        )

        return LLMResponse(
            content=text,
            model=active_model,
            usage=usage,
            finish_reason=finish_reason,
        )

    async def generate_embeddings(
        self,
        texts: list[str],
        *,
        dimensions: int | None = None,
        task_type: EmbeddingTaskType | None = None,
    ) -> list[list[float]]:
        """Generate embeddings using a stable Gemini embedding model."""
        if not texts:
            return []

        config: dict[str, object] | None = None
        if dimensions is not None:
            config = {}
            # Requests server-side projection when supported by the model.
            config["output_dimensionality"] = dimensions
        if task_type is not None:
            if config is None:
                config = {}
            config["task_type"] = task_type

        # For Gemini Embedding 2 Preview, passing a plain list[str] can be interpreted
        # as a single content. Build an explicit list[Content] to force one embedding
        # per input text.
        contents = [
            types.Content(parts=[types.Part.from_text(text=text)])
            for text in texts
        ]

        result = None

        for attempt in range(1, self.embedding_max_retries + 1):
            await self._wait_for_embedding_slot()
            try:
                result = await self.client.aio.models.embed_content(
                    model=self.embedding_model,
                    contents=contents,
                    config=config,
                )
                break
            except Exception as exc:
                can_retry_same_model = (
                    self._should_retry_embedding_error(exc)
                    and attempt < self.embedding_max_retries
                )

                if can_retry_same_model:
                    delay = self._extract_retry_delay_seconds(exc) or (
                        self.embedding_base_retry_seconds * (2 ** (attempt - 1))
                    )
                    # Small jitter avoids synchronized retries.
                    delay = min(delay, 30.0) + random.uniform(0.0, 0.5)

                    logger.warning(
                        "Embedding request failed on '%s' (attempt %d/%d). Retrying in %.2fs: %s",
                        self.embedding_model,
                        attempt,
                        self.embedding_max_retries,
                        delay,
                        type(exc).__name__,
                    )
                    await asyncio.sleep(delay)
                    continue

                raise RuntimeError(
                    f"Embedding failed on '{self.embedding_model}': "
                    f"{self._format_exception_summary(exc)}"
                ) from None

        if result is None:
            raise RuntimeError(
                f"No embeddings returned by model '{self.embedding_model}'"
            )

        embeddings = [emb.values for emb in result.embeddings]
        if dimensions is None and embeddings and self._cached_embedding_dimension is None:
            self._cached_embedding_dimension = len(embeddings[0])

        return embeddings

    async def _wait_for_embedding_slot(self) -> None:
        """Simple global rate limiter for embedding requests."""
        if self._embed_min_interval_seconds <= 0:
            return

        async with self._embed_lock:
            now = time.monotonic()
            elapsed = now - self._last_embed_request_at
            wait_seconds = self._embed_min_interval_seconds - elapsed
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            self._last_embed_request_at = time.monotonic()

    @staticmethod
    def _should_retry_embedding_error(exc: Exception) -> bool:
        """Return True for transient errors like 429/5xx and quota throttling."""
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int) and status_code in {429, 500, 502, 503, 504}:
            return True

        message = str(exc).lower()
        retry_markers = [
            "resource_exhausted",
            "quota exceeded",
            "rate limit",
            "too many requests",
            "temporarily unavailable",
            "service unavailable",
            "timeout",
        ]
        return any(marker in message for marker in retry_markers)

    @staticmethod
    def _extract_retry_delay_seconds(exc: Exception) -> float | None:
        """Parse provider-provided retry delays from error messages when available."""
        message = str(exc)

        patterns = [
            r"retry in\s+([\d.]+)s",
            r"retrydelay['\"]?:\s*['\"]?([\d.]+)s",
        ]
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return None
        return None

    @staticmethod
    def _format_exception_summary(exc: Exception) -> str:
        """Return a concise error summary without traceback details."""
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            return f"status={status_code} {type(exc).__name__}: {exc}"
        return f"{type(exc).__name__}: {exc}"

    async def get_embedding_dimension(self) -> int:
        """Return embedding dimension for the configured embedding model."""
        if self._cached_embedding_dimension is not None:
            return self._cached_embedding_dimension

        probe = await self.generate_embeddings(["dimension_probe"])
        if not probe or not probe[0]:
            raise RuntimeError(
                f"Could not determine embedding dimension for model '{self.embedding_model}'"
            )

        self._cached_embedding_dimension = len(probe[0])
        return self._cached_embedding_dimension

    @staticmethod
    def _normalize_model_list(models: Sequence[str] | None) -> list[str]:
        """Return a clean model list with empty values removed."""
        if not models:
            return []
        return [m.strip() for m in models if m and m.strip()]

    @staticmethod
    def _build_model_chain(primary: str, fallbacks: Sequence[str] | None = None) -> list[str]:
        """Build model chain preserving order and removing duplicates."""
        chain = [primary, *(fallbacks or [])]
        deduplicated: list[str] = []
        seen: set[str] = set()
        for model in chain:
            normalized = model.strip()
            if not normalized or normalized in seen:
                continue
            deduplicated.append(normalized)
            seen.add(normalized)
        return deduplicated
