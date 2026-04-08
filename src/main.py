"""
Main FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.api.routes import router
from src.rag.retriever_setup import load_vectorstore
from src.rag.reAct_agent import rebuild_agent
from src.db.sqlite_client import init_db, close_db
import os

<<<<<<< HEAD
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
=======

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # --- STARTUP ---
    # If a vectorstore was saved from a previous session, load it and
    # rebuild the agent so queries work immediately without re-uploading.
    from src.rag.retriever_setup import _faiss_vectorstore
    if _faiss_vectorstore is not None:
        print("Existing vector store found - rebuilding agent with persisted documents")
        from src.rag.reAct_agent import rebuild_agent
        rebuild_agent()
    else:
        print("No persisted vector store found - please upload a document")

    yield  # app runs here

    # --- SHUTDOWN (nothing needed) ---

>>>>>>> 4d7dd3b0bf130cc298fd25873b49c7895111969e

app = FastAPI(title="Adaptive RAG API", lifespan=lifespan)
app.include_router(router)
app.state.description_ = ""

@app.get("/")
async def root():
    """Root endpoint to verify API is running."""
    return {"message": "Adaptive RAG API is running"}