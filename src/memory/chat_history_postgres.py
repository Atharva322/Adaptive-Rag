import json
from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from src.db.postgres_client import get_pool

class PostgresChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in PostgreSQL"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
    
    async def add_message(self, message: BaseMessage) -> None:
        """Add a message to the session history"""
        msg_dict = message_to_dict(message)
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO chat_history (session_id, type, content, additional_kwargs)
                   VALUES ($1, $2, $3, $4)""",
                self.session_id,
                msg_dict["type"],
                msg_dict["data"]["content"],
                json.dumps(msg_dict["data"].get("additional_kwargs", {}))
            )
    
    async def get_messages(self) -> List[BaseMessage]:
        """Retrieve all messages for the session"""
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT type, content, additional_kwargs FROM chat_history 
                   WHERE session_id = $1 ORDER BY timestamp ASC""",
                self.session_id
            )
        
        # Convert to LangChain message format
        messages_data = []
        for row in rows:
            messages_data.append({
                "type": row["type"],
                "data": {
                    "content": row["content"],
                    "additional_kwargs": json.loads(row["additional_kwargs"]) if isinstance(row["additional_kwargs"], str) else row["additional_kwargs"]
                }
            })
        
        return messages_from_dict(messages_data)
    
    async def clear(self) -> None:
        """Clear all messages for the session"""
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM chat_history WHERE session_id = $1",
                self.session_id
            )

class ChatHistory:
    """Singleton chat history manager"""
    
    @staticmethod
    def get_session_history(session_id: str) -> PostgresChatMessageHistory:
        return PostgresChatMessageHistory(session_id)