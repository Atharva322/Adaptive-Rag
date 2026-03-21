"""
Retriever setup and vector store configuration.
"""

import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.core.config import settings

embeddings = OpenAIEmbeddings()

VECTORSTORE_PATH = "./vector_stores/faiss_index"

# Global variable - populated either from disk (on startup) or after upload
_faiss_vectorstore = None


def _try_load_from_disk():
    """Attempt to load vectorstore from disk."""
    global _faiss_vectorstore
    if os.path.exists(VECTORSTORE_PATH):
        try:
            _faiss_vectorstore = FAISS.load_local(
                folder_path=VECTORSTORE_PATH, 
                embeddings=embeddings, 
                allow_dangerous_deserialization=True
            )
            print(f"✓ Vector store loaded from disk: {VECTORSTORE_PATH}")
        except Exception as e:
            print(f"⚠ Could not load vector store from disk: {e}")
            _faiss_vectorstore = None
    else:
        print("ℹ No persistent vector store found on disk")


def get_retriever():
    """
    Get a retriever tool connected to the FAISS vector store.

    Returns the retriever tool that can search documents stored by retriever_chain().
    If no documents have been uploaded yet, creates a retriever with a dummy document.

    Returns:
        A LangChain retriever tool configured for the vector store.

    Raises:
        Exception: If vector store initialization fails.
    """
    global _faiss_vectorstore

    try:
        if _faiss_vectorstore is not None:
            retriever = _faiss_vectorstore.as_retriever()
            print("Using existing FAISS vectorstore with uploaded documents")
        else:
            print("No documents uploaded yet, creating dummy vectorstore")
            dummy_doc = Document(
                page_content="No documents have been uploaded yet. Please upload a document first.",
                metadata={"source": "initialization"}
            )
            _faiss_vectorstore = FAISS.from_documents(
                documents=[dummy_doc],
                embedding=embeddings
            )
            retriever = _faiss_vectorstore.as_retriever()

        # Load document description
        if os.path.exists("description.txt"):
            with open("description.txt", "r", encoding="utf-8") as f:
                description = f.read()
        else:
            description = None

        retriever_tool = create_retriever_tool(
            retriever,
            "retriever_customer_uploaded_documents",
            f"Use this tool **only** to answer questions about: {description}\n"
            "Don't use this tool to answer anything else."
        )

        return retriever_tool

    except Exception as e:
        print(f"Error initializing retriever: {e}")
        raise Exception(e)


def save_vectorstore(vectorstore):
    """Save FAISS vectorstore to disk."""
    try:
        Path(VECTORSTORE_PATH).parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(VECTORSTORE_PATH)
        print(f"✓ Vector store saved to {VECTORSTORE_PATH}")
    except Exception as e:
        print(f"⚠ Error saving vector store: {e}")


def load_vectorstore():
    """Load FAISS vectorstore from disk."""
    global _faiss_vectorstore
    if os.path.exists(VECTORSTORE_PATH):
        try:
            _faiss_vectorstore = FAISS.load_local(
                folder_path=VECTORSTORE_PATH, 
                embeddings=embeddings, 
                allow_dangerous_deserialization=True
            )
            print(f"✓ Vector store loaded from {VECTORSTORE_PATH}")
            return _faiss_vectorstore
        except Exception as e:
            print(f"⚠ Error loading vector store: {e}")
            return None
    return None

def add_documents_to_store(chunks: list):
    """Add documents to existing vectorstore without replacing it"""
    global _faiss_vectorstore
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    if _faiss_vectorstore is None:
        # Create new if doesn't exist
        _faiss_vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    else:
        # Append to existing vectorstore
        _faiss_vectorstore.add_documents(chunks)
    
    save_vectorstore(_faiss_vectorstore)
    return _faiss_vectorstore