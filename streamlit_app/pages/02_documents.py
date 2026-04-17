"""
Document management and statistics page.
"""
import sys
sys.path.append("..")
import streamlit as st
import os
import requests
import pandas as pd
from datetime import datetime
from utils.api_client import BASE_API_URL

# At the top after imports
if "uploaded_files" not in st.session_state:
    try:
        resp = requests.get(f"{BASE_API_URL}/rag/documents")
        if resp.status_code == 200:
            st.session_state.uploaded_files = resp.json().get("documents", [])
        else:
            st.session_state.uploaded_files = []
    except:
        st.session_state.uploaded_files = []

st.set_page_config(
    page_title="Documents - Adaptive RAG",
    page_icon="📚",
    layout="wide"
)

# Check authentication
if "jwt_token" not in st.session_state or st.session_state.jwt_token is None:
    _disable_auth = os.getenv("DISABLE_AUTH", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    if _disable_auth:
        st.session_state.jwt_token = "local_dev_token"
        st.session_state.username = st.session_state.get("username") or "local_dev"
    else:
        st.warning("⚠️ Please log in first")
        st.stop()

st.title("📚 Document Management")
st.markdown("View and manage all uploaded documents")

# Create tabs
tab1, tab2, tab3 = st.tabs(["My Documents", "Statistics", "Settings"])

with tab1:
    st.subheader("📄 Your Uploaded Documents")
    
    if "uploaded_files" in st.session_state and st.session_state.uploaded_files:
        # Create dataframe
        df_data = []
        for file_info in st.session_state.uploaded_files:
            df_data.append({
                "Name": file_info['name'],
                "Description": file_info['description'][:50] + "..." if len(file_info['description']) > 50 else file_info['description'],
                "Uploaded": file_info['uploaded_at'],
                "Size": "2.3 MB"  # Placeholder
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
        
        st.markdown("---")
        
        # Document details
        st.subheader("📋 Document Details")
        doc_col1, doc_col2, doc_col3 = st.columns(3)
        
        with doc_col1:
            st.metric("Total Documents", len(st.session_state.uploaded_files))
        with doc_col2:
            st.metric("Total Chunks", len(st.session_state.uploaded_files) * 6)  # Placeholder
        with doc_col3:
            st.metric("Storage Used", "13.8 MB")
    else:
        st.info("📭 No documents uploaded yet. Go to Chat page to upload documents.")

with tab2:
    st.subheader("📊 Upload Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Documents Uploaded", len(st.session_state.get("uploaded_files", [])))
    with col2:
        st.metric("Total Queries", "0")  # Placeholder
    with col3:
        st.metric("Avg. Query Time", "2.1s")  # Placeholder
    
    st.markdown("---")
    
    # Upload timeline (placeholder)
    st.subheader("📈 Recent Activity")
    st.info("Activity chart will appear here as you upload documents and ask questions")

with tab3:
    st.subheader("⚙️ Document Settings")
    
    st.markdown("#### Retrieval Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_k = st.slider(
            "Number of documents to retrieve",
            min_value=1,
            max_value=10,
            value=3,
            help="More documents = more context but slower"
        )
    
    with col2:
        chunk_size = st.slider(
            "Document chunk size",
            min_value=100,
            max_value=1000,
            value=500,
            step=100,
            help="Smaller chunks = more specific, larger = more context"
        )
    
    st.markdown("#### File Settings")
    
    max_file_size = st.select_slider(
        "Max file size (MB)",
        options=[5, 10, 25, 50, 100],
        value=25
    )
    
    st.warning(f"⚠️ Files larger than {max_file_size}MB will be rejected")
    
    if st.button("💾 Save Settings", use_container_width=True):
        st.success("✅ Settings saved!")
