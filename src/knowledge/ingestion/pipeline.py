"""Document ingestion pipeline — loads, chunks, embeds, and stores documents."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.knowledge.ingestion.chunker import TextChunker
from src.knowledge.ingestion.loaders import load_document

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
    """Orchestrates the full ingestion flow: load → chunk → embed → store."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_provider: LLMProvider,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embedding_batch_size: int = 20,
    ) -> None:
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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
        doc_id = hashlib.md5(str(path.absolute()).encode()).hexdigest()  # noqa: S324

        logger.info("Ingesting '%s' (type=%s, id=%s)", doc_title, doc_type, doc_id)

        text = load_document(path)
        if not text.strip():
            logger.warning("Empty document: %s", path)
            return 0

        chunks = self.chunker.chunk(text)
        logger.info("Created %d chunks from '%s'", len(chunks), doc_title)

        if not chunks:
            return 0

        doc_chunks = [
            DocumentChunk(
                id=f"{doc_id}_{i}",
                content=chunk,
                metadata={
                    "document_id": doc_id,
                    "document_title": doc_title,
                    "source_type": doc_type,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )
            for i, chunk in enumerate(chunks)
            if chunk.strip()
        ]

        if not doc_chunks:
            logger.warning("No non-empty chunks produced for '%s'", doc_title)
            return 0

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
            except Exception:
                logger.exception(
                    "Embedding batch failed for '%s' (chunks %d-%d). Skipping this batch and continuing.",
                    doc_title,
                    batch_start,
                    batch_start + len(batch) - 1,
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
                    except Exception:
                        logger.exception("Failed to embed chunk '%s'", chunk.id)
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
        raw_allowed = extensions or [".pdf", ".txt", ".md"]
        allowed = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in raw_allowed
        }
        total = 0

        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in allowed:
                try:
                    count = await self.ingest_file(path)
                except Exception:
                    logger.exception("Failed to ingest '%s'", path)
                    continue
                total += count

        logger.info("Directory ingest complete: %d total chunks", total)
        return total
