"""Abstract interface for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.config.settings import DEFAULT_LLM_MAX_TOKENS, DEFAULT_LLM_TEMPERATURE


@dataclass
class LLMResponse:
    """Standardised response from any LLM provider."""

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    raw: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Contract for LLM providers (Gemini, OpenAI, etc.)."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    ) -> LLMResponse:
        """Generate a completion from the model."""
        ...

    @abstractmethod
    async def generate_embeddings(
        self,
        texts: list[str],
        dimensions: int | None = None,
    ) -> list[list[float]]:
        """Generate embedding vectors for a list of texts."""
        ...

    @abstractmethod
    async def get_embedding_dimension(self) -> int:
        """Return the native output dimension of the active embedding model."""
        ...
