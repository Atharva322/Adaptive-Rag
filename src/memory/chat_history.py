"""
Chat history backend selector.

Local/dev runs often don't have Postgres configured. This module selects a
working async chat history backend based on environment configuration.
"""

from __future__ import annotations

import os

from src.memory.chat_history_postgres import ChatHistory as PostgresChatHistory
from src.memory.chat_history_async_memory import ChatHistory as MemoryChatHistory


class ChatHistory:
    """Chat history factory that selects Postgres when configured, else SQLite."""

    @staticmethod
    def get_session_history(session_id: str):
        if os.getenv("DATABASE_URL"):
            return PostgresChatHistory.get_session_history(session_id)
        # Prefer SQLite when available, otherwise fall back to in-memory.
        try:
            from src.memory.chat_history_sqlite import ChatHistory as SQLiteChatHistory  # noqa: WPS433

            return SQLiteChatHistory.get_session_history(session_id)
        except Exception:
            return MemoryChatHistory.get_session_history(session_id)
