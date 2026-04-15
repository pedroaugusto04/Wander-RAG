"""Qdrant vector store implementation."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.knowledge.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store using the async client."""

    def __init__(
        self,
        host: str = "qdrant",
        port: int = 6333,
        collection_name: str = "documents",
        vector_size: int = 768,
    ) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = AsyncQdrantClient(host=host, port=port)

    async def initialize(self) -> None:
        """Create collection if it doesn't exist."""
        collections = await self.client.get_collections()
        existing = {c.name for c in collections.collections}

        if self.collection_name not in existing:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "Created Qdrant collection '%s' (dim=%d)",
                self.collection_name,
                self.vector_size,
            )
        else:
            info = await self.client.get_collection(self.collection_name)
            existing_size = self._extract_vector_size(info)
            if existing_size is not None and existing_size != self.vector_size:
                logger.warning(
                    (
                        "Qdrant collection '%s' has dim=%d, configured=%d. "
                        "Adopting existing collection dimension to avoid mismatch."
                    ),
                    self.collection_name,
                    existing_size,
                    self.vector_size,
                )
                self.vector_size = existing_size
            logger.info(
                "Qdrant collection '%s' already exists (dim=%s)",
                self.collection_name,
                existing_size if existing_size is not None else "unknown",
            )

    @staticmethod
    def _extract_vector_size(info: Any) -> int | None:
        """Extract vector size from collection info for unnamed vector configs."""
        vectors_cfg = getattr(info.config.params, "vectors", None)
        if vectors_cfg is None:
            return None
        if hasattr(vectors_cfg, "size"):
            return int(vectors_cfg.size)
        if isinstance(vectors_cfg, dict):
            first = next(iter(vectors_cfg.values()), None)
            if first is not None and hasattr(first, "size"):
                return int(first.size)
        return None

    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert or update points in the collection."""
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)),
                vector=embedding,
                payload={"content": document, **metadata},
            )
            for doc_id, embedding, document, metadata in zip(
                ids, embeddings, documents, metadatas, strict=True
            )
        ]

        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.info("Upserted %d points into '%s'", len(points), self.collection_name)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        score_threshold: float = 0.3,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors with optional metadata filtering."""
        query_filter = None
        if filter_metadata:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_metadata.items()
            ]
            query_filter = Filter(must=conditions)

        results = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
        )

        return [
            {
                "content": point.payload.get("content", "") if point.payload else "",
                "score": point.score if point.score else 0.0,
                "metadata": {
                    k: v
                    for k, v in (point.payload or {}).items()
                    if k != "content"
                },
            }
            for point in results.points
        ]

    async def delete_by_document_id(self, document_id: str) -> None:
        """Delete all chunks belonging to a specific document."""
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )
        logger.info("Deleted chunks for document '%s'", document_id)

    async def get_collection_info(self) -> dict[str, Any]:
        """Return collection metadata."""
        info = await self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "status": info.status.value if info.status else "unknown",
        }
