"""
Main FastAPI application entry point.
"""
from src.rag.retriever_setup import _try_load_from_disk, load_vectorstore
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.api.routes import router
from src.rag.retriever_setup import load_vectorstore
from src.rag.reAct_agent import rebuild_agent
from src.db import postgres_client
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    close_history_db = None
    print("Initializing chat history database...")
    try:
        if os.getenv("DATABASE_URL"):
            await postgres_client.init_db()
            close_history_db = postgres_client.close_db
            print("[OK] PostgreSQL chat history initialized")
        else:
            # SQLite is optional (depends on `aiosqlite`). If it's not installed,
            # we still run with an in-memory chat history fallback.
            try:
                import aiosqlite  # noqa: F401
                from src.db import sqlite_client

                await sqlite_client.init_db()
                close_history_db = sqlite_client.close_db
                print("[OK] SQLite chat history initialized")
            except Exception:
                print("[INFO] SQLite unavailable; using in-memory chat history")
    except Exception as e:
        print(f"WARNING: Chat history initialization failed: {e}. Chat history may be unavailable.")

    print("Loading vectorstore from disk...")
    try:
        _try_load_from_disk()
        vectorstore = load_vectorstore()

        print("Rebuilding agent with loaded vectorstore...")
        rebuild_agent()
        print("✓ Agent rebuilt")
    except Exception as e:
        print(f"WARNING: Agent initialization failed: {e}. RAG features may be unavailable.")

    yield

    # Shutdown
    if close_history_db is not None:
        await close_history_db()
    print("[OK] Database connections closed")

app = FastAPI(title="Adaptive RAG API", lifespan=lifespan)
app.include_router(router)
app.state.description_ = ""
# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint to verify API is running."""
    return {"message": "Adaptive RAG API is running"}
