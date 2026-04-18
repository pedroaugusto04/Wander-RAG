"""Document ingestion pipeline — loads, cleans, chunks, embeds, and stores documents."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config.settings import (
    DEFAULT_LLAMA_PARSE_TIER,
    DEFAULT_RAG_CHUNK_OVERLAP,
    DEFAULT_RAG_CHUNK_SIZE,
    DEFAULT_RAG_EMBEDDING_BATCH_SIZE,
    DEFAULT_RAG_SUPPORTED_EXTENSIONS,
)
from src.knowledge.ingestion.chunker import MarkdownChunker
from src.knowledge.ingestion.loaders import DocumentLoader
from src.knowledge.ingestion.markdown_cleaner import MarkdownCleaner

if TYPE_CHECKING:
    from src.ai.llm.base import LLMProvider
    from src.knowledge.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document ready for embedding."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class IngestionPipeline:
    """Orchestrates the full ingestion flow: load → clean → chunk → embed → store."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_provider: LLMProvider,
        chunk_size: int = DEFAULT_RAG_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_RAG_CHUNK_OVERLAP,
        embedding_batch_size: int = DEFAULT_RAG_EMBEDDING_BATCH_SIZE,
        llama_api_key: str | None = None,
        llama_parse_tier: str = DEFAULT_LLAMA_PARSE_TIER,
    ) -> None:
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.loader = DocumentLoader(
            llama_api_key=llama_api_key,
            llama_parse_tier=llama_parse_tier,
        )
        self.cleaner = MarkdownCleaner()
        self.chunker = MarkdownChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_batch_size = max(1, embedding_batch_size)

    async def ingest_file(
        self,
        file_path: Path,
        document_title: str | None = None,
        source_type: str | None = None,
    ) -> int:
        """Ingest a single file into the vector store.

        Returns the number of chunks created.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        doc_title = document_title or path.stem
        doc_type = source_type or path.suffix.lstrip(".")
        doc_id = hashlib.sha256(str(path.absolute()).encode()).hexdigest()

        logger.info("Ingesting '%s' (type=%s, id=%s)", doc_title, doc_type, doc_id)

        # ── Step 1: Load ─────────────────────────────────────────────
        raw_text = await self.loader.load(path)
        if not raw_text.strip():
            logger.warning("Empty document: %s", path)
            return 0

        # ── Step 2: Clean Markdown ────────────────────────────────────
        cleaned = self.cleaner.clean(raw_text, source_filename=path.name)
        if not cleaned.strip():
            logger.warning("Document empty after cleaning: %s", path)
            return 0

        logger.debug(
            "Cleaned '%s': %d → %d chars",
            doc_title,
            len(raw_text),
            len(cleaned),
        )

        # ── Step 3: Chunk ─────────────────────────────────────────────
        chunk_contexts = self.chunker.chunk(cleaned)
        logger.info("Created %d chunks from '%s'", len(chunk_contexts), doc_title)

        if not chunk_contexts:
            return 0

        doc_chunks = [
            DocumentChunk(
                id=f"{doc_id}_{ctx.chunk_index}",
                content=ctx.content,
                metadata={
                    "document_id": doc_id,
                    "document_title": doc_title,
                    "source_type": doc_type,
                    "chunk_index": ctx.chunk_index,
                    "total_chunks": len(chunk_contexts),
                    "section_breadcrumb": ctx.section_breadcrumb,
                    "heading_level": ctx.heading_level,
                },
            )
            for ctx in chunk_contexts
            if ctx.content.strip()
        ]

        if not doc_chunks:
            logger.warning("No non-empty chunks produced for '%s'", doc_title)
            return 0

        # ── Step 4: Embed & Store ─────────────────────────────────────
        all_ids: list[str] = []
        all_embeddings: list[list[float]] = []
        all_documents: list[str] = []
        all_metadatas: list[dict[str, Any]] = []

        for batch_start in range(0, len(doc_chunks), self.embedding_batch_size):
            batch = doc_chunks[batch_start : batch_start + self.embedding_batch_size]
            texts = [c.content for c in batch]

            target_dimensions = getattr(self.vector_store, "vector_size", None)
            try:
                embeddings = await self.llm_provider.generate_embeddings(
                    texts,
                    dimensions=target_dimensions,
                )
            except Exception as exc:
                logger.error(
                    "Embedding batch failed for '%s' (chunks %d-%d). Skipping. Cause: %s",
                    doc_title,
                    batch_start,
                    batch_start + len(batch) - 1,
                    exc,
                )
                continue

            if len(embeddings) != len(batch):
                logger.warning(
                    "Embedding batch size mismatch for '%s': expected %d, got %d. "
                    "Falling back to per-chunk embedding.",
                    doc_title,
                    len(batch),
                    len(embeddings),
                )

                for chunk in batch:
                    try:
                        single = await self.llm_provider.generate_embeddings(
                            [chunk.content],
                            dimensions=target_dimensions,
                        )
                    except Exception as exc:
                        logger.error("Failed to embed chunk '%s': %s", chunk.id, exc)
                        continue

                    if not single:
                        logger.warning("No embedding returned for chunk '%s'", chunk.id)
                        continue

                    all_ids.append(chunk.id)
                    all_embeddings.append(single[0])
                    all_documents.append(chunk.content)
                    all_metadatas.append(chunk.metadata)
                continue

            for chunk, embedding in zip(batch, embeddings, strict=True):
                all_ids.append(chunk.id)
                all_embeddings.append(embedding)
                all_documents.append(chunk.content)
                all_metadatas.append(chunk.metadata)

        if not all_ids:
            logger.warning("No chunks with embeddings were generated for '%s'", doc_title)
            return 0

        # Deleta os chunks antigos deste documento para evitar chunks "fantasmas" (órfãos)
        # caso o documento atualizado tenha menos chunks que a versão anterior.
        try:
            if hasattr(self.vector_store, "delete_by_document_id"):
                await self.vector_store.delete_by_document_id(doc_id)
        except Exception:
            logger.warning("Failed to delete existing chunks for '%s'. May leave orphaned chunks.", doc_id)

        await self.vector_store.upsert(
            ids=all_ids,
            embeddings=all_embeddings,
            documents=all_documents,
            metadatas=all_metadatas,
        )

        logger.info(
            "Successfully ingested '%s': %d chunks stored",
            doc_title,
            len(doc_chunks),
        )
        return len(doc_chunks)

    async def ingest_directory(self, directory: Path, extensions: list[str] | None = None) -> int:
        """Ingest supported files recursively in a directory. Returns total chunks."""
        raw_allowed = extensions or DEFAULT_RAG_SUPPORTED_EXTENSIONS.split(",")
        allowed = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in raw_allowed
        }
        total = 0

        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in allowed:
                try:
                    count = await self.ingest_file(path)
                except Exception as exc:
                    logger.error("Failed to ingest '%s': %s", path, exc)
                    continue
                total += count

        logger.info("Directory ingest complete: %d total chunks", total)
        return total
