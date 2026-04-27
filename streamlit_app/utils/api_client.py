"""
API client for backend communication.
"""

import requests
import streamlit as st
import json

import os

# Use environment variable for backend URL (Render) or default to localhost
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
# Add https:// prefix if it's a Render URL without protocol
if BACKEND_URL and not BACKEND_URL.startswith("http"):
    BACKEND_URL = f"https://{BACKEND_URL}"

BASE_API_URL = BACKEND_URL
AUTH_API_URL = f"{BACKEND_URL}/api"

def get_api_token():
    """Get API token for authentication."""
    try:
        response = requests.get(f"{AUTH_API_URL}/init", timeout=5)
        return response.json()
    except Exception as e:
        st.error(f"Failed to get API token: {str(e)}")
        return {}

def create_user(email: str, password: str):
    """Create a new user account."""
    try:
        response = requests.post(
            f"{AUTH_API_URL}/signup",
            json={"email": email, "password": password},
            timeout=10
        )
        return response.json()
    except Exception as e:
        st.error(f"Signup error: {str(e)}")
        return {"success": False}

def login_user(email: str, password: str):
    """Login user and get JWT token."""
    try:
        response = requests.post(
            f"{AUTH_API_URL}/login",
            json={"email": email, "password": password},
            timeout=10
        )
        return response.json()
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        return {}

def query_backend(query: str, session_id: str):
    """Send RAG query to backend."""
    try:
        response = requests.post(
            f"{BASE_API_URL}/rag/query",
            json={
                "query": query,
                "session_id": session_id
            },
            headers={
                "accept": "application/json",
                "Content-Type": "application/json"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Backend error: {response.status_code}")
            return {}
            
    except Exception as e:
        st.error(f"Query error: {str(e)}")
        return {}

def document_upload_rag(file, description: str, session_id: str):
    """Upload document to RAG system."""
    try:
        files = {
            'file': (file.name, file.getvalue(), file.type)
        }
        
        response = requests.post(
            f"{BASE_API_URL}/rag/documents/upload",
            files=files,
            headers={
                "X-Description": description
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            # Return actual error message from backend
            try:
                error_detail = response.json().get("detail", response.text)
            except:
                error_detail = response.text or f"HTTP {response.status_code}"
            print(f"Upload error response: {error_detail}")  # Print to terminal too
            return {"status": "error", "message": error_detail}
            
    except Exception as e:
        error_msg = str(e)
        print(f"Upload exception: {error_msg}")  # Print to terminal
        return {"status": "error", "message": error_msg}
    
def delete_document(document_name: str):
    """Delete a document from the RAG system."""
    try:
        response = requests.delete(
            f"{BASE_API_URL}/rag/documents/{document_name}",
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "message": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def evaluate_ragas(
    question: str,
    ground_truth: str,
    include_per_sample: bool = True,
    metrics: list[str] | None = None,
):
    """Evaluate a single question-answer flow with RAGAS."""
    try:
        payload = {
            "dataset": [
                {
                    "question": question,
                    "ground_truth": ground_truth,
                }
            ],
            "include_per_sample": include_per_sample,
        }
        if metrics:
            payload["metrics"] = metrics

        response = requests.post(
            f"{BASE_API_URL}/rag/evaluate",
            json=payload,
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=120,
        )
        if response.status_code == 200:
            return {"status": "success", "data": response.json()}

        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text or f"HTTP {response.status_code}"
        return {"status": "error", "message": detail}
    except Exception as e:
        return {"status": "error", "message": str(e)}
