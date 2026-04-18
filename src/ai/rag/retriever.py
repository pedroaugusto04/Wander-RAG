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

_STOPWORDS = {
    "a",
    "ao",
    "aos",
    "as",
    "com",
    "como",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "ou",
    "para",
    "por",
    "qual",
    "quais",
    "que",
    "se",
    "sua",
    "suas",
    "seu",
    "seus",
    "um",
    "uma",
}


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

        # Even without a reranker, fetch extra candidates so lexical/hybrid
        # rescoring can recover better chunks from noisy institutional PDFs.
        fetch_k = k * self.retrieval_multiplier if self.reranker else max(
            k,
            k * max(2, self.retrieval_multiplier),
        )
        # Keep the dense stage slightly permissive and let the second pass
        # remove noisy candidates.
        search_threshold = 0.0 if self.reranker else max(0.0, threshold - 0.08)

        target_dimensions = getattr(self.vector_store, "vector_size", None)
        query_variants = self._build_query_variants(query)
        embeddings = await self.llm_provider.generate_embeddings(
            query_variants,
            dimensions=target_dimensions,
        )
        merged_results: dict[tuple[str, str, str], dict[str, Any]] = {}

        for idx, (variant, embedding) in enumerate(zip(query_variants, embeddings, strict=True)):
            variant_top_k = fetch_k if idx == 0 else min(fetch_k, max(k, 6))
            results = await self.vector_store.search(
                query_embedding=embedding,
                top_k=variant_top_k,
                score_threshold=search_threshold,
                filter_metadata=filter_metadata,
            )

            for result in results:
                metadata = result.get("metadata", {})
                key = self._result_key(result)
                previous = merged_results.get(key)
                if previous is None or float(result["score"]) > float(previous["score"]):
                    merged_results[key] = {
                        **result,
                        "metadata": {
                            **metadata,
                            "matched_query_variant": variant,
                        },
                    }

        chunks = [
            RetrievedChunk(
                content=result["content"],
                score=result["score"],
                metadata=result.get("metadata", {}),
            )
            for result in merged_results.values()
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
            chunks = self._apply_hybrid_rescoring(
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
            chunks = self._apply_hybrid_rescoring(
                query=query,
                chunks=chunks,
                top_k=k,
                list_intent=list_intent,
            )
            logger.info(
                "Retrieved %d chunks for query (threshold=%.3f, top score: %.3f): %s",
                len(chunks),
                threshold,
                float(chunks[0].metadata.get("vector_score", chunks[0].score))
                if chunks
                else 0.0,
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

    def _apply_hybrid_rescoring(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
        list_intent: bool,
    ) -> list[RetrievedChunk]:
        """Blend dense retrieval with lightweight lexical signals."""
        rescored: list[RetrievedChunk] = []

        for chunk in chunks:
            lexical_score = self._lexical_match_score(query=query, chunk=chunk)
            artifact_penalty = self._artifact_penalty(chunk)
            list_bonus = 0.0
            if list_intent:
                list_bonus = max(
                    0.0,
                    self._list_query_rank_key(query=query, chunk=chunk)[0] - chunk.score,
                )

            vector_score = float(chunk.metadata.get("vector_score", chunk.score))
            hybrid_score = (
                (vector_score * 0.72)
                + (lexical_score * 0.28)
                + list_bonus
                - artifact_penalty
            )

            metadata = {
                **chunk.metadata,
                "vector_score": vector_score,
                "lexical_score": lexical_score,
                "hybrid_score": hybrid_score,
                "artifact_penalty": artifact_penalty,
            }

            # Drop obvious parsing artefacts when they only survive by dense score.
            if artifact_penalty >= 0.14 and lexical_score < 0.18 and vector_score < 0.82:
                continue

            rescored.append(
                RetrievedChunk(
                    content=chunk.content,
                    score=hybrid_score,
                    metadata=metadata,
                )
            )

        rescored.sort(
            key=lambda chunk: (
                chunk.score,
                float(chunk.metadata.get("vector_score", chunk.score)),
                float(chunk.metadata.get("lexical_score", 0.0)),
            ),
            reverse=True,
        )

        if list_intent:
            rescored = self._select_final_chunks(
                query=query,
                chunks=rescored,
                top_k=top_k,
                list_intent=True,
            )
        else:
            rescored = rescored[:top_k]

        return rescored

    def _list_query_rank_key(
        self,
        *,
        query: str,
        chunk: RetrievedChunk,
    ) -> tuple[float, float]:
        """Boost chunks that are structurally suited for list answers."""
        rerank_score = float(chunk.metadata.get("hybrid_score", chunk.score))
        vector_score = float(chunk.metadata.get("vector_score", chunk.score))
        bonus = 0.0

        breadcrumb = str(chunk.metadata.get("section_breadcrumb", "")).lower()
        content = chunk.content

        if "lista" in breadcrumb:
            bonus += 0.12
        if self._looks_like_list_chunk(content):
            bonus += 0.1
        if self._asks_for_people(query) and self._contains_multiple_people_signals(content):
            bonus += 0.06

        return (rerank_score + bonus, vector_score)

    @classmethod
    def _build_query_variants(cls, query: str) -> list[str]:
        """Create a few lexical variants to improve recall on institutional jargon."""
        variants = [query.strip()]
        normalized = cls._normalize_query(query)

        replacements = (
            ("professores", "docentes"),
            ("professor", "docente"),
            ("professores do curso", "docentes de engenharia de computacao timoteo"),
            ("docentes", "corpo docente"),
            ("matriz curricular", "grade curricular"),
            (
                "matriz curricular",
                "quadro 13 matriz curricular engenharia de computacao timoteo",
            ),
            ("grade curricular", "matriz curricular"),
            ("secretaria", "secretaria de registro e controle academico"),
            ("atividades complementares", "ac oac atividades complementares"),
            ("trancamento de disciplina", "trancamento parcial de disciplina"),
        )

        for source, target in replacements:
            pattern = rf"\b{re.escape(source)}\b"
            if not re.search(pattern, normalized):
                continue
            variant = re.sub(pattern, target, normalized)
            if variant and variant not in variants:
                variants.append(variant)
            if len(variants) >= 3:
                break

        return variants

    @classmethod
    def _lexical_match_score(cls, *, query: str, chunk: RetrievedChunk) -> float:
        """Estimate lexical alignment between query terms and chunk surface text."""
        query_terms = cls._query_terms(query)
        if not query_terms:
            return 0.0

        title = str(chunk.metadata.get("document_title", ""))
        breadcrumb = str(chunk.metadata.get("section_breadcrumb", ""))
        surface = cls._normalize_query(f"{title} {breadcrumb} {chunk.content}")
        title_surface = cls._normalize_query(f"{title} {breadcrumb}")
        normalized_query = cls._normalize_query(query)

        overlap = sum(1 for term in query_terms if term in surface)
        title_overlap = sum(1 for term in query_terms if term in title_surface)
        coverage = overlap / len(query_terms)
        exact_phrase_bonus = 0.08 if normalized_query in surface else 0.0
        title_bonus = min(0.2, title_overlap * 0.08)

        phrase_bonus = 0.0
        phrase_markers = (
            "matriz curricular",
            "grade curricular",
            "quadro 13",
            "docentes",
            "professores",
            "atividades complementares",
            "trancamento",
        )
        for marker in phrase_markers:
            if marker in normalized_query and marker in title_surface:
                phrase_bonus += 0.12
        if "matriz curricular" in normalized_query and "quadro 13" in surface:
            phrase_bonus += 0.12

        return min(1.0, coverage + exact_phrase_bonus + title_bonus + phrase_bonus)

    @classmethod
    def _query_terms(cls, query: str) -> list[str]:
        """Return informative tokens only."""
        terms = re.findall(r"[a-z0-9à-ÿ]+", cls._normalize_query(query))
        return [term for term in terms if len(term) >= 3 and term not in _STOPWORDS]

    @staticmethod
    def _result_key(result: dict[str, Any]) -> tuple[str, str, str]:
        """Create a stable deduplication key for multi-query retrieval."""
        metadata = result.get("metadata", {})
        return (
            str(metadata.get("document_id", "")),
            str(metadata.get("chunk_index", "")),
            result.get("content", ""),
        )

    @staticmethod
    def _artifact_penalty(chunk: RetrievedChunk) -> float:
        """Penalize chunks that look like layout artefacts rather than content."""
        markers = (
            "layout attribution",
            "critical",
            "column ",
            "lista de abreviaturas e siglas",
        )
        title = str(chunk.metadata.get("document_title", "")).lower()
        breadcrumb = str(chunk.metadata.get("section_breadcrumb", "")).lower()
        surface = f"{title} {breadcrumb}"

        penalty = 0.0
        if any(marker in surface for marker in markers):
            penalty += 0.12
        if (
            title == "matriz"
            or "matriz curricular" in surface
            or "quadro 13" in surface
        ):
            penalty = max(0.0, penalty - 0.12)
        if title == "matriz" and len(chunk.content.strip()) < 80:
            penalty += 0.04
        return penalty

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
