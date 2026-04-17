"""
Async in-memory chat history storage.

Used as a safe local fallback when no database is configured/available.
"""

from __future__ import annotations

from typing import Dict, List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage


class AsyncInMemoryChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, store: Dict[str, List[BaseMessage]]):
        self.session_id = session_id
        self._store = store

    async def add_message(self, message: BaseMessage) -> None:
        self._store.setdefault(self.session_id, []).append(message)

    async def get_messages(self) -> List[BaseMessage]:
        return list(self._store.get(self.session_id, []))

    async def clear(self) -> None:
        self._store.pop(self.session_id, None)


class ChatHistory:
    """Async in-memory chat history factory."""

    _store: Dict[str, List[BaseMessage]] = {}

    @classmethod
    def get_session_history(cls, session_id: str) -> AsyncInMemoryChatMessageHistory:
        return AsyncInMemoryChatMessageHistory(session_id=session_id, store=cls._store)

