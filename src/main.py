"""
Main FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

from src.api.routes import router

app = FastAPI(title="Adaptive RAG API")
app.include_router(router)
app.state.description_ = ""

@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup: if disk has a saved vectorstore, rebuild the agent with it
    from src.rag.retriever_setup import _faiss_vectorstore
    if _faiss_vectorstore is not None:
        print("Existing vector store found - rebuilding agent")
        from src.rag.reAct_agent import rebuild_agent
        rebuild_agent()
    yield

@app.get("/")
async def root():
    """Root endpoint to verify API is running."""
    return {"message": "Adaptive RAG API is running"}
