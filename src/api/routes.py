"""
API routes for RAG operations.
"""

from fastapi import APIRouter, UploadFile, File, Header
from httpcore import request
from langchain_core.messages import HumanMessage, AIMessage

from src.memory.chat_history import ChatHistory
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

    await chat_history.add_message(HumanMessage(content=req.query))

    result = builder.invoke(
        {"messages": [HumanMessage(content=req.query)]},
        config={"recursion_limit": 50}
    )
    
    initial_state = {
    "messages": [HumanMessage(content=req.query)],
    "latest_query": req.query,
    "rewrite_count": 0,
    "metadata_filter": req.metadata_filter,   # NEW
}

    if result.get("messages"):
        last_message = result["messages"][-1]
        if hasattr(last_message, 'content'):
            await chat_history.add_message(AIMessage(content=last_message.content))
        elif isinstance(last_message, dict):
            await chat_history.add_message(AIMessage(content=last_message.get("content", "")))

    return result

@router.get("/rag/chat_history")
async def get_chat_history(session_id: str):
    """Retrieve chat history for a session."""
    chat_history = ChatHistory.get_session_history(session_id)
    messages = await chat_history.get_messages()

    history = []
    for msg in messages:
        history.append({
            "role": "user" if msg.type == "human" else "assistant",
            "content": msg.content
        })

    return {"messages": history}


@router.post("/rag/documents/upload")
async def upload_file(
    file: UploadFile = File(...),
    description: str = Header(..., alias="X-Description")
):
    """Upload a document for RAG processing."""
    status_upload = documents(description, file)

    rebuild_agent()

    return status_upload


@router.delete("/rag/documents/{document_name}")
async def delete_document(document_name: str):
    """Delete a document from the RAG system."""
    success = delete_document_from_store(document_name)
    if success:
        return {"status": "success", "message": f"Document '{document_name}' deleted"}
    return {"status": "error", "message": f"Document '{document_name}' not found"}
