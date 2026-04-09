"""
ChromaDB client configuration.
"""

import chromadb
from chromadb.config import Settings
import os

# Persistent storage path
PERSIST_DIRECTORY = "./vector_stores/chroma"

def get_chroma_client():
    """
    Get a persistent ChromaDB client.
    
    Returns:
        chromadb.Client: Configured ChromaDB client with persistent storage.
    """
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    
    client = chromadb.PersistentClient(
        path=PERSIST_DIRECTORY,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    return client

# Global client instance
_chroma_client = None

def initialize_chroma():
    """Initialize the global ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = get_chroma_client()
        print(f"✓ ChromaDB initialized at {PERSIST_DIRECTORY}")
    return _chroma_client