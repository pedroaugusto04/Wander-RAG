"""CLI script to ingest documents into the vector store."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add project root to sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def main() -> None:
    """Ingest documents from a directory or single file."""
    import argparse

    from src.ai.llm.gemini_provider import GeminiProvider
    from src.config.settings import get_settings
    from src.infra.logging import setup_logging
    from src.knowledge.ingestion.pipeline import IngestionPipeline
    from src.knowledge.vectorstore.qdrant_store import QdrantVectorStore

    settings = get_settings()

    # Configura o log para aparecer no console durante o script
    setup_logging(log_level=settings.app_log_level)

    parser = argparse.ArgumentParser(description="Ingest documents into Wander Jr knowledge base")
    parser.add_argument(
        "path",
        nargs="?",
        type=str,
        default=settings.rag_documents_path,
        help="Path to a file or directory to ingest (default: RAG_DOCUMENTS_PATH)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Document title (only for single files)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override chunk size (default from settings)",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=None,
        help="File extensions to process (default: RAG_SUPPORTED_EXTENSIONS)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show collection info and exit",
    )
    parser.add_argument(
        "--no-llamaparse",
        action="store_true",
        help="Force pypdf fallback (skip LlamaParse even if API key is set)",
    )

    args = parser.parse_args()

    # Initialize components
    llm_provider = GeminiProvider(
        api_key=settings.gemini_api_key,
        model=settings.llm_model,
        fallback_models=settings.llm_fallback_model_list,
        embedding_model=settings.embedding_model,
        embedding_fallback_models=settings.embedding_fallback_model_list,
        embedding_requests_per_minute=settings.embedding_requests_per_minute,
        embedding_max_retries=settings.embedding_max_retries,
        embedding_base_retry_seconds=settings.embedding_base_retry_seconds,
    )

    detected_embedding_dim = await llm_provider.get_embedding_dimension()
    vector_size = settings.embedding_dimensions or detected_embedding_dim

    vector_store = QdrantVectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection_name,
        vector_size=vector_size,
        sparse_vector_name=settings.qdrant_sparse_vector_name,
    )
    await vector_store.initialize()

    if args.info:
        info = await vector_store.get_collection_info()
        print(f"\n📊 Collection: {info['name']}")
        print(f"   Points: {info['points_count']}")
        print(f"   Status: {info['status']}")
        return

    llama_api_key = None if args.no_llamaparse else settings.llama_cloud_api_key

    pipeline = IngestionPipeline(
        vector_store=vector_store,
        llm_provider=llm_provider,
        chunk_size=args.chunk_size or settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        embedding_batch_size=settings.rag_embedding_batch_size,
        llama_api_key=llama_api_key,
        llama_parse_tier=settings.llama_parse_tier,
    )

    target = Path(args.path)

    if target.is_file():
        print(f"\n📄 Ingesting file: {target.name}")
        count = await pipeline.ingest_file(target, document_title=args.title)
        print(f"✅ Done! Created {count} chunks.")
    elif target.is_dir():
        print(f"\n📁 Ingesting directory: {target}")
        count = await pipeline.ingest_directory(
            target,
            extensions=args.extensions or settings.rag_supported_extensions_list,
        )
        print(f"✅ Done! Created {count} total chunks.")
    else:
        print(f"❌ Path not found: {target}", file=sys.stderr)
        sys.exit(1)

    # Show collection info after ingestion
    info = await vector_store.get_collection_info()
    print(f"\n📊 Collection '{info['name']}': {info['points_count']} total points")


if __name__ == "__main__":
    asyncio.run(main())
