import aiosqlite
import os
from pathlib import Path

# SQLite database path
DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/adaptive_rag.db")

# Ensure data directory exists
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

async def get_db():
    """Get SQLite database connection"""
    conn = await aiosqlite.connect(DB_PATH)
    conn.row_factory = aiosqlite.Row
    return conn

async def init_db():
    """Initialize database schema"""
    conn = await get_db()
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            additional_kwargs TEXT DEFAULT '{}',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_id ON chat_history(session_id)
    """)
    await conn.commit()
    await conn.close()

async def close_db():
    """Close database connection (placeholder for compatibility)"""
    pass