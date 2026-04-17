"""RAG Pipeline — coordinates retrieval and retrieval-query preparation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.ai.rag.retriever import RAGRetriever
from src.config.settings import (
    DEFAULT_APP_ASSISTANT_NAME,
    DEFAULT_APP_INSTITUTION_NAME,
    DEFAULT_RAG_CONFIDENCE_LOW_THRESHOLD,
    DEFAULT_RAG_CONFIDENCE_NONE_THRESHOLD,
    DEFAULT_RAG_LIST_QUERY_MIN_TOP_K,
    DEFAULT_RAG_PROMPT_HISTORY_TURNS,
    DEFAULT_RAG_SCORE_THRESHOLD,
    DEFAULT_RAG_TOP_K,
)

if TYPE_CHECKING:
    from src.ai.llm.base import LLMProvider
    from src.ai.rag.reranker import FlashRankReranker
    from src.core.models import RetrievedChunk
    from src.knowledge.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """High-level RAG pipeline: retrieve context → build prompt → return."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_provider: LLMProvider,
        top_k: int = DEFAULT_RAG_TOP_K,
        score_threshold: float = DEFAULT_RAG_SCORE_THRESHOLD,
        confidence_none_threshold: float = DEFAULT_RAG_CONFIDENCE_NONE_THRESHOLD,
        confidence_low_threshold: float = DEFAULT_RAG_CONFIDENCE_LOW_THRESHOLD,
        reranker: FlashRankReranker | None = None,
        retrieval_multiplier: int = 3,
        list_query_min_top_k: int = DEFAULT_RAG_LIST_QUERY_MIN_TOP_K,
        assistant_name: str = DEFAULT_APP_ASSISTANT_NAME,
        institution_name: str = DEFAULT_APP_INSTITUTION_NAME,
        prompt_history_turns: int = DEFAULT_RAG_PROMPT_HISTORY_TURNS,
    ) -> None:
        self.retriever = RAGRetriever(
            vector_store=vector_store,
            llm_provider=llm_provider,
            top_k=top_k,
            score_threshold=score_threshold,
            reranker=reranker,
            retrieval_multiplier=retrieval_multiplier,
            list_query_min_top_k=list_query_min_top_k,
        )
        self.confidence_none_threshold = confidence_none_threshold
        self.confidence_low_threshold = max(
            confidence_low_threshold,
            confidence_none_threshold,
        )
        self.assistant_name = assistant_name
        self.institution_name = institution_name
        self.prompt_history_turns = max(1, prompt_history_turns)

    async def process(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Run the RAG pipeline for a user query.

        Returns:
            {
                "chunks": [...],             # Retrieved chunks
                "retrieved_chunks": [...],   # Prompt-ready chunk dicts
                "confidence": "high"|"low"|"none",
                "max_score": float,
                "retrieval_query": str,
            }
        """
        retrieval_query = await self._rewrite_query(query, conversation_history)
        chunks: list[RetrievedChunk] = await self.retriever.retrieve(retrieval_query)

        # Prefer dense retrieval score for confidence when available.
        max_score = max(
            (
                float(c.metadata.get("vector_score", c.score))
                for c in chunks
            ),
            default=0.0,
        )
        if not chunks or max_score < self.confidence_none_threshold:
            confidence = "none"
        elif max_score < self.confidence_low_threshold:
            confidence = "low"
        else:
            confidence = "high"

        chunk_dicts = [
            {
                "content": c.content,
                "source": self._build_chunk_source(c.metadata),
            }
            for c in chunks
        ]

        return {
            "chunks": chunks,
            "retrieved_chunks": chunk_dicts,
            "confidence": confidence,
            "max_score": max_score,
            "retrieval_query": retrieval_query,
        }

    @staticmethod
    def _build_chunk_source(metadata: dict[str, Any]) -> str:
        """Build a short human-readable source label for prompt citations."""
        title = metadata.get("document_title", "Documento institucional")
        breadcrumb = str(metadata.get("section_breadcrumb", "")).strip()
        if breadcrumb:
            return f"{title} — {breadcrumb}"
        return str(title)

    async def _rewrite_query(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None,
    ) -> str:
        """Rewrite follow-up questions into standalone retrieval queries."""
        if not conversation_history:
            return query

        try:
            from src.ai.rag.prompts import build_query_rewrite_prompt

            messages = build_query_rewrite_prompt(
                user_question=query,
                conversation_history=conversation_history,
                assistant_name=self.assistant_name,
                max_history_turns=self.prompt_history_turns,
            )
            response = await self.retriever.llm_provider.generate(
                messages=messages,
                temperature=0.0,
                max_tokens=96,
            )
        except Exception as exc:
            logger.warning("Query rewrite failed, using original query: %s", exc)
            return query

        rewritten = " ".join(response.content.splitlines()).strip().strip("\"'")
        return rewritten or query
