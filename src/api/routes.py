"""
API routes for RAG operations.
"""

from fastapi import APIRouter, UploadFile, File, Header
from langchain_core.messages import HumanMessage

from src.memory.chat_history_postgres import ChatHistory
from src.models.query_request import QueryRequest
from src.rag.document_upload import documents
from src.rag.graph_builder import builder
from src.rag.retriever_setup import load_document_metadata, delete_document_from_store
from src.rag.reAct_agent import rebuild_agent

router = APIRouter()

@router.get("/rag/documents")
async def list_documents():
    """Return list of all uploaded documents."""
    return {"documents": load_document_metadata()}


@router.post("/rag/query")
async def rag_query(req: QueryRequest):
    """Process a RAG query and return the result."""
    chat_history = ChatHistory.get_session_history(req.session_id)
    result = builder.invoke(
        {"messages": [HumanMessage(content=req.query)]},
        config={"recursion_limit": 50}
    )
    return result


@router.post("/rag/documents/upload")
async def upload_file(
    file: UploadFile = File(...),
    description: str = Header(..., alias="X-Description")
):
    """Upload a document for RAG processing."""
    status_upload = documents(description, file)

    # Rebuild the agent so it uses the newly uploaded document
    rebuild_agent()
    
    return status_upload


@router.delete("/rag/documents/{document_name}")
async def delete_document(document_name: str):
    """Delete a document from the RAG system."""
    success = delete_document_from_store(document_name)
    if success:
        rebuild_agent()
        return {"status": "success", "message": f"Document '{document_name}' deleted"}
    return {"status": "error", "message": f"Document '{document_name}' not found"}