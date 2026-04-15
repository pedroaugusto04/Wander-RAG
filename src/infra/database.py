"""Database connection management for PostgreSQL."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def create_db_engine(database_url: str) -> async_sessionmaker[AsyncSession]:
    """Create an async SQLAlchemy engine and session factory.

    Args:
        database_url: PostgreSQL connection string
            (e.g. postgresql+asyncpg://user:pass@host:5432/db)
    """
    engine = create_async_engine(
        database_url,
        echo=False,
        pool_size=5,
        max_overflow=10,
    )

    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
