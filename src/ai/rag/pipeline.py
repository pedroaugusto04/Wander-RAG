"""RAG Pipeline — coordinates retrieval and prompt building."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.ai.rag.prompts import build_rag_prompt
from src.ai.rag.retriever import RAGRetriever

if TYPE_CHECKING:
    from src.ai.llm.base import LLMProvider
    from src.core.models import RetrievedChunk
    from src.knowledge.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """High-level RAG pipeline: retrieve context → build prompt → return."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_provider: LLMProvider,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> None:
        self.retriever = RAGRetriever(
            vector_store=vector_store,
            llm_provider=llm_provider,
            top_k=top_k,
            score_threshold=score_threshold,
        )

    async def process(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Run the RAG pipeline for a user query.

        Returns:
            {
                "messages": [...],           # Prompt messages ready for LLM
                "chunks": [...],             # Retrieved chunks
                "confidence": "high"|"low"|"none",
                "max_score": float,
            }
        """
        # Retrieve relevant chunks
        chunks: list[RetrievedChunk] = await self.retriever.retrieve(query)

        # Determine confidence level
        max_score = max((c.score for c in chunks), default=0.0)
        if not chunks or max_score < 0.3:
            confidence = "none"
        elif max_score < 0.6:
            confidence = "low"
        else:
            confidence = "high"

        # Build prompt
        chunk_dicts = [
            {
                "content": c.content,
                "source": c.metadata.get("document_title", "Documento institucional"),
            }
            for c in chunks
        ]

        messages = build_rag_prompt(
            user_question=query,
            retrieved_chunks=chunk_dicts,
            conversation_history=conversation_history,
        )

        return {
            "messages": messages,
            "chunks": chunks,
            "confidence": confidence,
            "max_score": max_score,
        }
