"""
Retriever setup and vector store configuration with ChromaDB.
"""

import os
from pathlib import Path
import json
from typing import List

from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.rag.hybrid_retriever import (
    build_bm25_retriever,
    clear_bm25_retriever,
    get_bm25_retriever,
    get_hybrid_retriever,
)

from src.db.chroma_client import initialize_chroma
import src.config.settings as rag_settings

embeddings = OpenAIEmbeddings()

# ChromaDB collection name
COLLECTION_NAME = "adaptive_rag_documents"

# Global variable - populated either from disk (on startup) or after upload
_chroma_vectorstore = None

METADATA_PATH = "./vector_stores/documents_metadata.json"

def save_document_metadata(name: str, description: str, doc_ids: list = []):
    """Append uploaded document metadata to persistent JSON file."""
    metadata = load_document_metadata()
    metadata.append({
        "name": name,
        "description": description,
        "uploaded_at": __import__('datetime').datetime.now().isoformat(),
        "doc_ids": doc_ids
    })
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f)

def load_document_metadata() -> list:
    """Load persisted document metadata list."""
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    return []


def _try_load_from_disk():
    """Attempt to load vectorstore from ChromaDB."""
    global _chroma_vectorstore
    try:
        chroma_client = initialize_chroma()
        _chroma_vectorstore = Chroma(
            client=chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings
        )
        
        # Check if collection has documents
        collection = chroma_client.get_collection(COLLECTION_NAME)
        count = collection.count()
        
        if count > 0:
            print(f"[OK] ChromaDB loaded with {count} documents")
            rebuild_bm25_from_store(k=rag_settings.RETRIEVER_K)
        else:
            print("[INFO] ChromaDB collection exists but is empty")
            clear_bm25_retriever()
            
    except Exception as e:
        print(f"[WARN] Could not load ChromaDB collection: {e}")
        _chroma_vectorstore = None


def get_retriever(
    search_type: str = "similarity",  # "similarity" | "mmr" | "hybrid"
    k: int = 4,
    metadata_filter: dict | None = None,
):
    vectorstore = load_vectorstore()
    search_kwargs = {"k": k}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    if search_type == "mmr":
        search_kwargs.update({"fetch_k": rag_settings.MMR_FETCH_K, "lambda_mult": rag_settings.MMR_LAMBDA_MULT})
        return vectorstore.as_retriever(search_type="mmr", search_kwargs=search_kwargs)

    vector_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)

    if search_type == "hybrid":
        if get_bm25_retriever() is None:
            rebuild_bm25_from_store(k=k)
        return get_hybrid_retriever(vector_retriever, rag_settings.BM25_WEIGHT, rag_settings.VECTOR_WEIGHT)

    return vector_retriever


def get_retriever_tool(
    search_type: str = "similarity",
    k: int = 4,
    metadata_filter: dict | None = None,
):
    """Return a LangChain tool backed by the configured retriever."""
    retriever = get_retriever(
        search_type=search_type,
        k=k,
        metadata_filter=metadata_filter,
    )
    return create_retriever_tool(
        retriever,
        "document_retriever",
        "Search and return relevant snippets from uploaded documents.",
    )


def load_vectorstore():
    """Load ChromaDB vectorstore."""
    global _chroma_vectorstore
    try:
        chroma_client = initialize_chroma()
        _chroma_vectorstore = Chroma(
            client=chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings
        )
        print(f"[OK] ChromaDB collection '{COLLECTION_NAME}' loaded")
        return _chroma_vectorstore
    except Exception as e:
        print(f"[WARN] Error loading ChromaDB: {e}")
        return None


def _get_all_docs_from_store() -> List[Document]:
    vectorstore = load_vectorstore()
    results = vectorstore._collection.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(results["documents"], results["metadatas"])
    ]


def rebuild_bm25_from_store(k: int | None = None) -> None:
    """Rebuild in-memory BM25 index from all docs currently in Chroma."""
    all_docs = _get_all_docs_from_store()
    if not all_docs:
        clear_bm25_retriever()
        return
    build_bm25_retriever(all_docs, k=k or rag_settings.RETRIEVER_K)

def add_documents_to_store(documents: List[Document]) -> list[str]:
    vectorstore = load_vectorstore()
    ids = vectorstore.add_documents(documents)
    rebuild_bm25_from_store(k=rag_settings.RETRIEVER_K)
    return ids

def delete_document_from_store(name: str) -> bool:
    """Remove a document from ChromaDB and metadata by filename."""
    global _chroma_vectorstore
    metadata = load_document_metadata()
    
    # Find the entry
    entry = next((m for m in metadata if m["name"] == name), None)
    if not entry:
        return False
    
    # Delete from ChromaDB if IDs stored
    doc_ids = entry.get("doc_ids", [])
    if _chroma_vectorstore and doc_ids:
        try:
            _chroma_vectorstore.delete(ids=doc_ids)
            print(f"[OK] Deleted {len(doc_ids)} documents from ChromaDB")
        except Exception as e:
            print(f"[WARN] Error deleting from ChromaDB: {e}")
    
    # Remove from metadata
    metadata = [m for m in metadata if m["name"] != name]
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f)

    rebuild_bm25_from_store(k=rag_settings.RETRIEVER_K)
    
    return True
