"""Abstract interface for vector store backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.config.settings import DEFAULT_RAG_SCORE_THRESHOLD, DEFAULT_RAG_TOP_K


class VectorStore(ABC):
    """Contract for vector store implementations (Qdrant, etc.)."""

    @abstractmethod
    async def initialize(self) -> None:
        """Create collection / ensure schema exists."""
        ...

    @abstractmethod
    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert or update vectors with their metadata."""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = DEFAULT_RAG_TOP_K,
        score_threshold: float = DEFAULT_RAG_SCORE_THRESHOLD,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors. Returns list of {content, score, metadata}."""
        ...

    @abstractmethod
    async def delete_by_document_id(self, document_id: str) -> None:
        """Delete all chunks belonging to a specific document."""
        ...

    @abstractmethod
    async def get_collection_info(self) -> dict[str, Any]:
        """Return metadata about the collection (count, etc.)."""
        ...
