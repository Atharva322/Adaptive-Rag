"""
State model for the graph-based RAG system.
"""

from typing import TypedDict, Annotated, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]
    binary_score: str
    route: str
    latest_query: str
    rewrite_count: int
    metadata_filter: Optional[dict]   # NEW

    
