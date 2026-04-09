"""
ChromaDB client for Railway-hosted server.
"""

import chromadb
import os
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

def get_chroma_client():
    """Get ChromaDB client connected to Railway server."""
    
    host = os.getenv('CHROMA_HOST')
    port_str = os.getenv('CHROMA_PORT')
    use_ssl_str = os.getenv('CHROMA_USE_SSL')

    # Fallback: parse RAILWAY_SERVICE_CHROMADB_SERVER_URL if CHROMA_HOST not set
    if not host:
        railway_chroma_url = os.getenv('RAILWAY_SERVICE_CHROMADB_SERVER_URL')
        if railway_chroma_url:
            parsed = urlparse(railway_chroma_url)
            host = parsed.hostname
            if parsed.port:
                port_str = str(parsed.port)
            elif parsed.scheme == 'https':
                port_str = '443'
            else:
                port_str = '80'
            use_ssl_str = 'true' if parsed.scheme == 'https' else 'false'
            print(f"Using RAILWAY_SERVICE_CHROMADB_SERVER_URL: {railway_chroma_url}")

    host = host or 'localhost'
    port = int(port_str or '8000')
    use_ssl = (use_ssl_str or 'false').lower() == 'true'

    print(f"ChromaDB Config: host={host}, port={port}, ssl={use_ssl}")

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
        print("✓ Connected to ChromaDB")
    return _chroma_client
