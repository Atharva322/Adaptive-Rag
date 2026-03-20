"""
Main FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import router


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


app = FastAPI(title="Adaptive RAG API", lifespan=lifespan)
app.include_router(router)
app.state.description_ = ""


@app.get("/")
async def root():
    """Root endpoint to verify API is running."""
    return {"message": "Adaptive RAG API is running"}
