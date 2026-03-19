"""
API routes for RAG operations.
"""

from fastapi import APIRouter, UploadFile, File, Header
from langchain_core.messages import HumanMessage

from src.memory.chat_history_mongo import ChatHistory
from src.models.query_request import QueryRequest
from src.rag.document_upload import documents
from src.rag.graph_builder import builder

router = APIRouter()


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
    from src.rag.reAct_agent import rebuild_agent
    rebuild_agent()

    return {"status": status_upload}
