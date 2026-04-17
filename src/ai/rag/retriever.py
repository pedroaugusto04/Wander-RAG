"""RAG retriever — searches the vector store, optionally reranks, and builds context."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from src.config.settings import (
    DEFAULT_RAG_LIST_QUERY_MIN_TOP_K,
    DEFAULT_RAG_SCORE_THRESHOLD,
    DEFAULT_RAG_TOP_K,
)
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
        top_k: int = DEFAULT_RAG_TOP_K,
        score_threshold: float = DEFAULT_RAG_SCORE_THRESHOLD,
        reranker: FlashRankReranker | None = None,
        retrieval_multiplier: int = 3,
        list_query_min_top_k: int = DEFAULT_RAG_LIST_QUERY_MIN_TOP_K,
    ) -> None:
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.reranker = reranker
        self.retrieval_multiplier = retrieval_multiplier
        self.list_query_min_top_k = max(1, list_query_min_top_k)

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
        requested_k = top_k if top_k is not None else self.top_k
        list_intent = self._is_list_query(query)
        k = self._effective_top_k(query, requested_k)
        threshold = score_threshold if score_threshold is not None else self.score_threshold

        # Over-fetch when a reranker is available.
        fetch_k = k * self.retrieval_multiplier if self.reranker else k
        # For two-stage retrieval, keep dense search permissive and let reranker decide.
        search_threshold = 0.0 if self.reranker else threshold

        target_dimensions = getattr(self.vector_store, "vector_size", None)
        embeddings = await self.llm_provider.generate_embeddings(
            [query],
            dimensions=target_dimensions,
        )
        query_embedding = embeddings[0]

        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=fetch_k,
            score_threshold=search_threshold,
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
            chunks = self.reranker.rerank(query=query, chunks=chunks, top_k=fetch_k)
            # Keep only passages that still satisfy the original dense similarity threshold.
            chunks = [
                c
                for c in chunks
                if float(c.metadata.get("vector_score", c.score)) >= threshold
            ]
            chunks = self._select_final_chunks(
                query=query,
                chunks=chunks,
                top_k=k,
                list_intent=list_intent,
            )
            logger.info(
                (
                    "Reranked %d→%d chunks (requested_top_k=%d, effective_top_k=%d, "
                    "list_intent=%s, threshold=%.3f, top rerank=%.3f): %s"
                ),
                fetch_k,
                len(chunks),
                requested_k,
                k,
                list_intent,
                threshold,
                chunks[0].score if chunks else 0.0,
                query[:80],
            )
        else:
            logger.info(
                "Retrieved %d chunks for query (threshold=%.3f, top score: %.3f): %s",
                len(chunks),
                threshold,
                chunks[0].score if chunks else 0.0,
                query[:80],
            )

        return chunks

    def _effective_top_k(self, query: str, requested_k: int) -> int:
        """Increase context budget for list-style questions that need more chunks."""
        if self._is_list_query(query):
            return max(requested_k, self.list_query_min_top_k)
        return requested_k

    def _select_final_chunks(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
        *,
        list_intent: bool,
    ) -> list[RetrievedChunk]:
        """Return the final chunk set after intent-aware rerank adjustments."""
        if not list_intent:
            return chunks[:top_k]

        scored = [
            (
                self._list_query_rank_key(query=query, chunk=chunk),
                idx,
                chunk,
            )
            for idx, chunk in enumerate(chunks)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _key, _idx, chunk in scored[:top_k]]

    def _list_query_rank_key(
        self,
        *,
        query: str,
        chunk: RetrievedChunk,
    ) -> tuple[float, float]:
        """Boost chunks that are structurally suited for list answers."""
        rerank_score = chunk.score
        vector_score = float(chunk.metadata.get("vector_score", chunk.score))
        bonus = 0.0

        breadcrumb = str(chunk.metadata.get("section_breadcrumb", "")).lower()
        content = chunk.content

        if "lista" in breadcrumb:
            bonus += 0.04
        if self._looks_like_list_chunk(content):
            bonus += 0.03
        if self._asks_for_people(query) and self._contains_multiple_people_signals(content):
            bonus += 0.02

        return (rerank_score + bonus, vector_score)

    @staticmethod
    def _is_list_query(query: str) -> bool:
        """Detect questions that ask for a list of people/items."""
        normalized = RAGRetriever._normalize_query(query)
        list_markers = (
            "quais",
            "quem sao",
            "quem são",
            "lista",
            "nomes",
        )
        people_or_items = (
            "professor",
            "professores",
            "docente",
            "docentes",
            "servidor",
            "servidores",
            "disciplinas",
            "itens",
            "documentos",
        )
        return (
            any(marker in normalized for marker in list_markers)
            and any(term in normalized for term in people_or_items)
        )

    @staticmethod
    def _asks_for_people(query: str) -> bool:
        """Detect when the list is specifically about people."""
        normalized = RAGRetriever._normalize_query(query)
        return any(
            term in normalized
            for term in ("professor", "professores", "docente", "docentes", "servidor")
        )

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Lowercase and collapse whitespace for simple intent heuristics."""
        return re.sub(r"\s+", " ", query.lower()).strip()

    @staticmethod
    def _looks_like_list_chunk(text: str) -> bool:
        """Return True for chunks that contain several bullet/list markers."""
        return len(re.findall(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+", text)) >= 2

    @staticmethod
    def _contains_multiple_people_signals(text: str) -> bool:
        """Approximate whether a chunk lists several names/emails."""
        email_count = text.count("@")
        dash_lines = len(re.findall(r"(?m)^\s*[-*+]\s+", text))
        return email_count >= 2 or dash_lines >= 2
