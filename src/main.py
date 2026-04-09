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
from src.db.postgres_client import init_db, close_db
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Initializing PostgreSQL database...")
    try:
        await init_db()
        print("✓ PostgreSQL database initialized")
    except Exception as e:
        print(f"WARNING: PostgreSQL initialization failed: {e}. Chat history features will be unavailable.")

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
    await close_db()
    print("✓ Database connections closed")

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
