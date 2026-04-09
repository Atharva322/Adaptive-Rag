"""
ChromaDB client for Railway-hosted server.
"""

import chromadb
import os

def get_chroma_client():
    """Get ChromaDB client connected to Railway server."""
    
    host = os.getenv('CHROMA_HOST', 'localhost')
    port = int(os.getenv('CHROMA_PORT', '8000'))
    use_ssl = os.getenv('CHROMA_USE_SSL', 'false').lower() == 'true'
    
    # DEBUG LOGGING - REMOVE AFTER TESTING
    print(f"🔍 DEBUG ChromaDB Config:")
    print(f"   CHROMA_HOST env var: {os.getenv('CHROMA_HOST')}")  
    print(f"   CHROMA_PORT env var: {os.getenv('CHROMA_PORT')}")
    print(f"   CHROMA_USE_SSL env var: {os.getenv('CHROMA_USE_SSL')}")
    print(f"   Resolved host: {host}")
    print(f"   Resolved port: {port}")
    print(f"   Resolved SSL: {use_ssl}")
    # END DEBUG
    
    client = chromadb.HttpClient(
        host=host,
        port=port,
        ssl=use_ssl
    )
    
    return client

_chroma_client = None

def initialize_chroma():
    """Initialize the global ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = get_chroma_client()
        print(f"✓ Connected to ChromaDB at {os.getenv('CHROMA_HOST')}")
    return _chroma_client