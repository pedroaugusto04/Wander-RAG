"""FlashRank reranker — lightweight ONNX-based cross-encoder for retrieval.

Uses FlashRank (no PyTorch dependency) to re-score query–passage pairs
after the initial vector search, improving retrieval precision.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.models import RetrievedChunk

logger = logging.getLogger(__name__)


class FlashRankReranker:
    """Re-ranks retrieved chunks using a cross-encoder model via FlashRank.

    Typical usage in a two-stage retrieval pipeline:
      1. Bi-encoder (Gemini Embedding) fetches ``top_k * multiplier`` candidates.
      2. This reranker re-scores and returns the best ``top_k``.
    """

    def __init__(
        self,
        model_name: str = "ms-marco-MultiBERT-L-12",
        top_k: int = 5,
    ) -> None:
        from flashrank import Ranker

        self.top_k = top_k
        self._ranker = Ranker(model_name=model_name)
        logger.info("FlashRank reranker loaded (model=%s)", model_name)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Re-rank *chunks* for *query* and return the best *top_k*.

        Each returned chunk preserves its original metadata and adds the
        original vector score as ``metadata["vector_score"]``.
        """
        if not chunks:
            return []

        from flashrank import RerankRequest

        from src.core.models import RetrievedChunk as _RetrievedChunk

        k = top_k or self.top_k

        passages = [
            {"id": str(i), "text": chunk.content}
            for i, chunk in enumerate(chunks)
        ]

        request = RerankRequest(query=query, passages=passages)
        results = self._ranker.rerank(request)

        reranked: list[RetrievedChunk] = []
        for result in results[:k]:
            original_idx = int(result["id"])
            original = chunks[original_idx]
            reranked.append(
                _RetrievedChunk(
                    content=original.content,
                    score=float(result["score"]),
                    metadata={
                        **original.metadata,
                        "vector_score": original.score,
                    },
                )
            )

        logger.debug(
            "Reranked %d → %d chunks (top rerank score: %.4f)",
            len(chunks),
            len(reranked),
            reranked[0].score if reranked else 0.0,
        )

        return reranked
