"""Qdrant vector store implementation."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVectorParams,
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
        collection_name: str = "documents2",
        vector_size: int = 768,
    ) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.dense_vector_name: str | None = None
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
                sparse_vectors_config={"sparse-vector": SparseVectorParams()},
            )
            logger.info(
                "Created Qdrant collection '%s' (dim=%d, sparse='sparse-vector')",
                self.collection_name,
                self.vector_size,
            )
        else:
            info = await self.client.get_collection(self.collection_name)
            existing_size, dense_vector_name = self._extract_dense_vector_info(info)
            self.dense_vector_name = dense_vector_name
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

            sparse_names = self._extract_sparse_vector_names(info)
            if sparse_names and "sparse-vector" not in sparse_names:
                logger.warning(
                    "Qdrant collection '%s' has sparse vectors %s; expected 'sparse-vector'.",
                    self.collection_name,
                    sorted(sparse_names),
                )

            logger.info(
                "Qdrant collection '%s' already exists (dim=%s, dense_name=%s, sparse=%s)",
                self.collection_name,
                existing_size if existing_size is not None else "unknown",
                self.dense_vector_name or "<default>",
                sorted(sparse_names) if sparse_names else "<none>",
            )

    @staticmethod
    def _extract_dense_vector_info(info: Any) -> tuple[int | None, str | None]:
        """Extract dense vector size and optional vector name from collection info."""
        vectors_cfg = getattr(info.config.params, "vectors", None)
        if vectors_cfg is None:
            return None, None
        if hasattr(vectors_cfg, "size"):
            return int(vectors_cfg.size), None
        if isinstance(vectors_cfg, Mapping):
            for name, cfg in vectors_cfg.items():
                if cfg is not None and hasattr(cfg, "size"):
                    normalized_name = str(name) if name else None
                    return int(cfg.size), normalized_name
        return None, None

    @staticmethod
    def _extract_sparse_vector_names(info: Any) -> set[str]:
        """Extract sparse vector names from collection info."""
        sparse_cfg = getattr(info.config.params, "sparse_vectors", None)
        if isinstance(sparse_cfg, Mapping):
            return {str(name) for name in sparse_cfg.keys() if name}
        return set()

    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert or update points in the collection."""
        def vector_payload(embedding: list[float]) -> list[float] | dict[str, list[float]]:
            if self.dense_vector_name:
                return {self.dense_vector_name: embedding}
            return embedding

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)),
                vector=vector_payload(embedding),
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
            using=self.dense_vector_name,
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
