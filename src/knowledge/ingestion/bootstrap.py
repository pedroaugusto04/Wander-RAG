"""Startup helpers for document ingestion."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.config.settings import DEFAULT_LLAMA_PARSE_TIER
from src.knowledge.ingestion.pipeline import IngestionPipeline

if TYPE_CHECKING:
    from pathlib import Path

    from src.ai.llm.base import LLMProvider
    from src.knowledge.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


async def auto_ingest_if_empty(
    *,
    vector_store: VectorStore,
    llm_provider: LLMProvider,
    documents_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    embedding_batch_size: int,
    llama_api_key: str | None = None,
    llama_parse_tier: str = DEFAULT_LLAMA_PARSE_TIER,
    extensions: list[str] | None = None,
) -> int:
    """Ingest documents only when the target collection has no points."""
    info = await vector_store.get_collection_info()
    points_count = int(info.get("points_count") or 0)

    if points_count > 0:
        logger.info(
            "Skipping auto-ingest: collection '%s' already has %d points",
            info.get("name", "unknown"),
            points_count,
        )
        return 0

    if not documents_path.exists():
        logger.warning("Skipping auto-ingest: documents path not found (%s)", documents_path)
        return 0

    if not documents_path.is_dir():
        logger.warning("Skipping auto-ingest: documents path is not a directory (%s)", documents_path)
        return 0

    logger.info(
        "Collection is empty. Auto-ingesting documents from '%s'...",
        documents_path,
    )

    pipeline = IngestionPipeline(
        vector_store=vector_store,
        llm_provider=llm_provider,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_batch_size=embedding_batch_size,
        llama_api_key=llama_api_key,
        llama_parse_tier=llama_parse_tier,
    )

    try:
        ingested = await pipeline.ingest_directory(documents_path, extensions=extensions)
    except Exception:
        logger.exception("Auto-ingest failed for '%s'", documents_path)
        return 0

    logger.info("Auto-ingest complete: %d chunks added", ingested)
    return ingested
