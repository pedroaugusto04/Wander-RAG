"""RAG retriever — searches the vector store and builds context for the LLM."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.core.models import RetrievedChunk

if TYPE_CHECKING:
    from src.ai.llm.base import LLMProvider
    from src.knowledge.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Retrieves relevant document chunks from the vector store."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_provider: LLMProvider,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> None:
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.top_k = top_k
        self.score_threshold = score_threshold

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Search for relevant chunks given a user query.

        1. Generates an embedding for the query
        2. Searches the vector store
        3. Filters by score threshold
        4. Returns ranked chunks
        """
        k = top_k or self.top_k
        threshold = score_threshold or self.score_threshold

        # Generate query embedding
        target_dimensions = getattr(self.vector_store, "vector_size", None)
        embeddings = await self.llm_provider.generate_embeddings(
            [query],
            dimensions=target_dimensions,
        )
        query_embedding = embeddings[0]

        # Search vector store
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=k,
            score_threshold=threshold,
            filter_metadata=filter_metadata,
        )

        chunks = [
            RetrievedChunk(
                content=r["content"],
                score=r["score"],
                metadata=r.get("metadata", {}),
            )
            for r in results
        ]

        logger.info(
            "Retrieved %d chunks for query (top score: %.3f): %s",
            len(chunks),
            chunks[0].score if chunks else 0.0,
            query[:80],
        )

        return chunks
