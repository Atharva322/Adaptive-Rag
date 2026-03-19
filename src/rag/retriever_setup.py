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

# Global variable to store the FAISS vectorstore instance
# This ensures get_retriever() can access documents stored by retriever_chain()
_faiss_vectorstore = None


def retriever_chain(chunks: list[Document]):
    """
    Initialize and store documents in FAISS vector database.

    Args:
        chunks: List of document chunks to store.

    Returns:
        Boolean indicating success of the operation.
    """
    global _faiss_vectorstore

    try:
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        # Store the vectorstore globally so get_retriever() can access it
        _faiss_vectorstore = vectorstore

        print("FAISS vector store initialized with documents")
        print(f"Vectorstore contains {len(chunks)} document chunks")
        return True
    except Exception as e:
        print(f"Error storing documents in FAISS: {e}")
        return False


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
    """Save FAISS vectorstore to disk after upload"""
    Path(VECTORSTORE_PATH).parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"Vector store saved to {VECTORSTORE_PATH}")


def load_vectorstore():
    """Load FAISS vectorstore from disk for queries"""
    if os.path.exists(VECTORSTORE_PATH):
        vs = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
        print(f"Vector store loaded from {VECTORSTORE_PATH}")
        return vs
    return None
