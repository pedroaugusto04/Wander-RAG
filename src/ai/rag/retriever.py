"""RAG retriever — searches the vector store, optionally reranks, and builds context."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.core.models import RetrievedChunk

if TYPE_CHECKING:
    from src.ai.llm.base import LLMProvider
    from src.ai.rag.reranker import FlashRankReranker
    from src.knowledge.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Retrieves relevant document chunks from the vector store.

    When a *reranker* is provided the retriever over-fetches candidates
    (``top_k * retrieval_multiplier``) and then re-scores them with a
    cross-encoder to return the final *top_k*.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        llm_provider: LLMProvider,
        top_k: int = 5,
        score_threshold: float = 0.3,
        reranker: FlashRankReranker | None = None,
        retrieval_multiplier: int = 3,
    ) -> None:
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.reranker = reranker
        self.retrieval_multiplier = retrieval_multiplier

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Search for relevant chunks given a user query.

        1. Generates an embedding for the query.
        2. Searches the vector store (over-fetches when reranker is active).
        3. Optionally reranks with a cross-encoder.
        4. Filters by score threshold.
        5. Returns ranked chunks.
        """
        k = top_k or self.top_k
        threshold = score_threshold or self.score_threshold

        # Over-fetch when a reranker is available.
        fetch_k = k * self.retrieval_multiplier if self.reranker else k

        target_dimensions = getattr(self.vector_store, "vector_size", None)
        embeddings = await self.llm_provider.generate_embeddings(
            [query],
            dimensions=target_dimensions,
        )
        query_embedding = embeddings[0]

        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=fetch_k,
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

        # ── Rerank stage ──────────────────────────────────────────────
        if self.reranker and chunks:
            chunks = self.reranker.rerank(query=query, chunks=chunks, top_k=k)
            logger.info(
                "Reranked %d→%d chunks for query (top rerank=%.3f): %s",
                fetch_k,
                len(chunks),
                chunks[0].score if chunks else 0.0,
                query[:80],
            )
        else:
            logger.info(
                "Retrieved %d chunks for query (top score: %.3f): %s",
                len(chunks),
                chunks[0].score if chunks else 0.0,
                query[:80],
            )

        return chunks
