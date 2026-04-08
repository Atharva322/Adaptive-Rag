"""
Main FastAPI application entry point.
"""

from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.api.routes import router
from src.rag.retriever_setup import load_vectorstore
from src.rag.reAct_agent import rebuild_agent
from src.db.sqlite_client import init_db, close_db
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Initializing SQLite database...")
    await init_db()
    print(f"✓ SQLite database initialized at: {os.getenv('SQLITE_DB_PATH', './data/adaptive_rag.db')}")
    
    print("Loading vectorstore from disk...")
    load_vectorstore()
    print("✓ Vector store loaded")
    
    print("Rebuilding agent with loaded vectorstore...")
    rebuild_agent()
    print("✓ Agent rebuilt")
    
    yield
    
    # Shutdown
    await close_db()
    print("✓ Database connections closed")

app = FastAPI(title="Adaptive RAG API", lifespan=lifespan)
app.include_router(router)
app.state.description_ = ""

@app.get("/")
async def root():
    """Root endpoint to verify API is running."""
    return {"message": "Adaptive RAG API is running"}