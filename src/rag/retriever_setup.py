"""
Retriever setup and vector store configuration with ChromaDB.
"""

import os
from pathlib import Path
import json

from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from src.db.chroma_client import initialize_chroma
from src.core.config import settings

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
            print(f"✓ ChromaDB loaded with {count} documents")
        else:
            print("ℹ ChromaDB collection exists but is empty")
            
    except Exception as e:
        print(f"⚠ Could not load ChromaDB collection: {e}")
        _chroma_vectorstore = None


def get_retriever():
    """
    Get a retriever tool connected to the ChromaDB vector store.

    Returns:
        A LangChain retriever tool configured for the vector store.
    """
    global _chroma_vectorstore

    try:
        if _chroma_vectorstore is not None:
            retriever = _chroma_vectorstore.as_retriever(search_kwargs={"k": 4})
            print("Using existing ChromaDB vectorstore with uploaded documents")
        else:
            print("No documents uploaded yet, creating empty ChromaDB collection")
            chroma_client = initialize_chroma()
            
            # Create collection with a dummy document
            dummy_doc = Document(
                page_content="No documents have been uploaded yet. Please upload a document first.",
                metadata={"source": "initialization"}
            )
            _chroma_vectorstore = Chroma.from_documents(
                documents=[dummy_doc],
                embedding=embeddings,
                client=chroma_client,
                collection_name=COLLECTION_NAME
            )
            retriever = _chroma_vectorstore.as_retriever(search_kwargs={"k": 4})

        # Load document description
        if os.path.exists("description.txt"):
            with open("description.txt", "r", encoding="utf-8") as f:
                description = f.read()
        else:
            description = "uploaded documents and knowledge base"

        retriever_tool = create_retriever_tool(
            retriever,
            "retriever_customer_uploaded_documents",
            f"Search through {description}. Use this tool to find relevant information from uploaded documents."
        )

        return retriever_tool

    except Exception as e:
        print(f"Error initializing retriever: {e}")
        raise Exception(e)


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
        print(f"✓ ChromaDB collection '{COLLECTION_NAME}' loaded")
        return _chroma_vectorstore
    except Exception as e:
        print(f"⚠ Error loading ChromaDB: {e}")
        return None


def add_documents_to_store(chunks: list) -> list:
    """Add documents to existing ChromaDB collection without replacing it"""
    global _chroma_vectorstore
    
    chroma_client = initialize_chroma()
    
    if _chroma_vectorstore is None:
        # Create new collection
        _chroma_vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=chroma_client,
            collection_name=COLLECTION_NAME
        )
    else:
        # Add to existing collection
        _chroma_vectorstore.add_documents(chunks)
    
    # Return document IDs
    collection = chroma_client.get_collection(COLLECTION_NAME)
    all_ids = collection.get()['ids']
    
    # Return the last N IDs (newly added)
    new_ids = all_ids[-len(chunks):] if len(all_ids) >= len(chunks) else all_ids
    
    print(f"✓ Added {len(chunks)} documents to ChromaDB")
    return new_ids


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
            print(f"✓ Deleted {len(doc_ids)} documents from ChromaDB")
        except Exception as e:
            print(f"⚠ Error deleting from ChromaDB: {e}")
    
    # Remove from metadata
    metadata = [m for m in metadata if m["name"] != name]
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f)
    
    return True