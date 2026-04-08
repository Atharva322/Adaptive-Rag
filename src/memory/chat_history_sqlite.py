import json
from datetime import datetime
from typing import List
import aiosqlite
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from src.db.sqlite_client import get_db, DB_PATH

class SQLiteChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in SQLite"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
    
    async def add_message(self, message: BaseMessage) -> None:
        """Add a message to the session history"""
        msg_dict = message_to_dict(message)
        conn = await get_db()
        await conn.execute(
            """INSERT INTO chat_history (session_id, type, content, additional_kwargs)
               VALUES (?, ?, ?, ?)""",
            (
                self.session_id,
                msg_dict["type"],
                msg_dict["data"]["content"],
                json.dumps(msg_dict["data"].get("additional_kwargs", {}))
            )
        )
        await conn.commit()
        await conn.close()
    
    async def get_messages(self) -> List[BaseMessage]:
        """Retrieve all messages for the session"""
        conn = await get_db()
        cursor = await conn.execute(
            """SELECT type, content, additional_kwargs FROM chat_history 
               WHERE session_id = ? ORDER BY timestamp ASC""",
            (self.session_id,)
        )
        rows = await cursor.fetchall()
        await conn.close()
        
        # Convert to LangChain message format
        messages_data = []
        for row in rows:
            messages_data.append({
                "type": row[0],
                "data": {
                    "content": row[1],
                    "additional_kwargs": json.loads(row[2])
                }
            })
        
        return messages_from_dict(messages_data)
    
    async def clear(self) -> None:
        """Clear all messages for the session"""
        conn = await get_db()
        await conn.execute(
            "DELETE FROM chat_history WHERE session_id = ?",
            (self.session_id,)
        )
        await conn.commit()
        await conn.close()

class ChatHistory:
    """Singleton chat history manager"""
    
    @staticmethod
    def get_session_history(session_id: str) -> SQLiteChatMessageHistory:
        return SQLiteChatMessageHistory(session_id)