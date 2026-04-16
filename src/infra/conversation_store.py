"""PostgreSQL-backed persistence for conversation sessions and turns."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.core.models import ConversationContext, MessageRole

logger = logging.getLogger(__name__)


class PostgresConversationStore:
    """Persist and load conversation data in PostgreSQL."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    async def initialize(self) -> None:
        """Create required tables and indexes if they do not exist."""
        create_sessions_sql = text(
            """
            CREATE TABLE IF NOT EXISTS conversation_sessions (
                session_id TEXT PRIMARY KEY,
                channel TEXT NOT NULL,
                channel_user_id TEXT NOT NULL,
                channel_chat_id TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_activity TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
            );
            """
        )

        create_turns_sql = text(
            """
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id BIGSERIAL PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
            );
            """
        )

        create_indexes_sql = [
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_turns_session_created
                ON conversation_turns(session_id, created_at DESC);
                """
            ),
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_sessions_last_activity
                ON conversation_sessions(last_activity DESC);
                """
            ),
        ]

        async with self.session_factory() as session:
            await session.execute(create_sessions_sql)
            await session.execute(create_turns_sql)
            for index_sql in create_indexes_sql:
                await session.execute(index_sql)
            await session.commit()

        logger.info("Conversation persistence initialized in PostgreSQL")

    async def upsert_session(
        self,
        context: ConversationContext,
        *,
        channel_chat_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create or update a session row."""
        payload = metadata or {}
        upsert_sql = text(
            """
            INSERT INTO conversation_sessions (
                session_id,
                channel,
                channel_user_id,
                channel_chat_id,
                created_at,
                last_activity,
                metadata_json
            )
            VALUES (
                :session_id,
                :channel,
                :channel_user_id,
                :channel_chat_id,
                :created_at,
                :last_activity,
                CAST(:metadata_json AS JSONB)
            )
            ON CONFLICT (session_id)
            DO UPDATE SET
                channel = EXCLUDED.channel,
                channel_user_id = EXCLUDED.channel_user_id,
                channel_chat_id = COALESCE(EXCLUDED.channel_chat_id, conversation_sessions.channel_chat_id),
                last_activity = EXCLUDED.last_activity,
                metadata_json = EXCLUDED.metadata_json;
            """
        )

        params = {
            "session_id": context.session_id,
            "channel": context.channel.value,
            "channel_user_id": context.channel_user_id,
            "channel_chat_id": channel_chat_id,
            "created_at": context.created_at,
            "last_activity": context.last_activity,
            "metadata_json": json.dumps(payload, ensure_ascii=False),
        }

        async with self.session_factory() as session:
            await session.execute(upsert_sql, params)
            await session.commit()

    async def append_turn(
        self,
        *,
        context: ConversationContext,
        role: MessageRole,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist a conversation turn."""
        insert_sql = text(
            """
            INSERT INTO conversation_turns (session_id, role, content, created_at, metadata_json)
            VALUES (:session_id, :role, :content, :created_at, CAST(:metadata_json AS JSONB));
            """
        )

        params = {
            "session_id": context.session_id,
            "role": role.value,
            "content": content,
            "created_at": datetime.now(UTC),
            "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
        }

        async with self.session_factory() as session:
            await session.execute(insert_sql, params)
            await session.commit()

    async def get_recent_history(self, session_id: str, *, max_turns: int) -> list[dict[str, str]]:
        """Load recent conversation turns ordered from oldest to newest."""
        query_sql = text(
            """
            SELECT role, content
            FROM (
                SELECT role, content, created_at
                FROM conversation_turns
                WHERE session_id = :session_id
                ORDER BY created_at DESC
                LIMIT :max_turns
            ) t
            ORDER BY created_at ASC;
            """
        )

        async with self.session_factory() as session:
            result = await session.execute(
                query_sql,
                {
                    "session_id": session_id,
                    "max_turns": max_turns,
                },
            )
            rows = result.fetchall()

        return [{"role": str(row.role), "content": str(row.content)} for row in rows]
