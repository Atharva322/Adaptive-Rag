"""
Main FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

from src.api.routes import router

app = FastAPI(title="Adaptive RAG API")
app.include_router(router)
app.state.description_ = ""
from src.api.routes import router
from contextlib import asynccontextmanager
from src.rag.retriever_setup import load_vectorstore
from src.rag.reAct_agent import rebuild_agent

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Loading vectorstore from disk...")
    load_vectorstore()
    print("Rebuilding agent with loaded vectorstore...")
    rebuild_agent()
    yield
    # Shutdown
    pass

app = FastAPI(lifespan=lifespan)
# Create app with lifespan, then include router
app = FastAPI(title="Adaptive RAG API", lifespan=lifespan)
app.include_router(router)
app.state.description_ = ""

@app.get("/")
async def root():
    """Root endpoint to verify API is running."""
    return {"message": "Adaptive RAG API is running"}
