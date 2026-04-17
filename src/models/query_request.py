"""
Query request model.
"""

from git import Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    metadata_filter: Optional[dict] = None   # NEW
    # e.g. {"doc_type": {"$eq": "pdf"}}