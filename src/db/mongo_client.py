import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

# Read from .env file instead of hardcoding
MONGO_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGODB_DB_NAME", "adaptive_rag")

print(f"Connecting to MongoDB: {MONGO_URL[:50]}...")  # Log for debugging

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

async def get_db():
    """Get database instance"""
    return db

async def close_db():
    """Close database connection"""
    client.close()