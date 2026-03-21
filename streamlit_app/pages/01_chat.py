"""
Main chat interface with RAG capabilities and source document display.
"""

import streamlit as st
import requests
import json
from datetime import datetime
import sys
sys.path.append("..")
from utils.api_client import query_backend, document_upload_rag

st.set_page_config(
    page_title="Adaptive RAG - Chat",
    page_icon="💬",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Check authentication
if "jwt_token" not in st.session_state or st.session_state.jwt_token is None:
    st.warning("⚠️ Please log in first")
    st.stop()

# Generate session ID if not exists
if not st.session_state.session_id:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# Header
col1, col2 = st.columns([0.9, 0.1])
with col1:
    st.title("💬 Adaptive RAG Chat")
    st.markdown("Ask questions about your uploaded documents")

with col2:
    if st.button("🚪 Logout"):
        st.session_state.jwt_token = None
        st.session_state.session_id = None
        st.rerun()

# Sidebar - Document Management
with st.sidebar:
    st.header("📄 Document Management")
    
    # File upload section
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file (PDF or TXT)",
        type=["pdf", "txt"],
        key=f"file_uploader_{len(st.session_state.uploaded_files)}"
    )
    
    if uploaded_file:
        doc_description = st.text_area(
            "Document Description",
            placeholder="Briefly describe what this document contains...",
            height=80
        )
        
        if st.button("📤 Upload Document", use_container_width=True):
            if not doc_description:
                st.error("Please provide a description for the document")
            else:
                with st.spinner("Uploading and processing..."):
                    try:
                        result = document_upload_rag(
                            file=uploaded_file,
                            description=doc_description,
                            session_id=st.session_state.session_id
                        )
                        
                        if result.get("status") == "success":
                            st.session_state.uploaded_files.append({
                                "name": uploaded_file.name,
                                "description": doc_description,
                                "uploaded_at": datetime.now().isoformat()
                            })
                            st.success(f"✅ {uploaded_file.name} uploaded successfully!")
                            st.rerun()
                            st.info(f"📊 Created {result.get('chunks', 0)} document chunks for retrieval")
                        else:
                            st.error(f"Upload failed: {result.get('message', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Error uploading document: {str(e)}")
    
    # Uploaded files list
    if st.session_state.uploaded_files:
        st.divider()
        st.subheader("✅ Uploaded Documents")
        for idx, file_info in enumerate(st.session_state.uploaded_files):
            with st.expander(f"📄 {file_info['name']}"):
                st.write(f"**Description:** {file_info['description']}")
                st.write(f"**Uploaded:** {file_info['uploaded_at']}")
                
                if st.button("🗑️ Delete", key=f"delete_{idx}", use_container_width=True):
                    # Note: You'll need to add a delete endpoint to your backend
                    st.session_state.uploaded_files.pop(idx)
                    st.success("Document removed from session")
    else:
        st.info("No documents uploaded yet. Upload a document to get started!")

# Main chat area
st.subheader("💭 Conversation")

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(message["content"])
            
            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources", expanded=False):
                    cols = st.columns(len(message["sources"]))
                    for idx, source in enumerate(message["sources"]):
                        with cols[idx]:
                            st.info(f"📄 {source}")
            
            # Display metadata if available
            if "metadata" in message:
                meta = message["metadata"]
                col1, col2 = st.columns(2)
                with col1:
                    if "confidence" in meta:
                        confidence = meta["confidence"]
                        st.metric("Confidence", f"{confidence*100:.0f}%")
                with col2:
                    if "processing_time" in meta:
                        st.metric("Processing Time", f"{meta['processing_time']:.2f}s")

# Chat input
st.divider()
user_input = st.chat_input(
    "Ask a question about your documents...",
    key="chat_input"
)

if user_input:
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Display user message immediately
    with st.chat_message("user", avatar="👤"):
        st.write(user_input)
    
    # Query backend
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Searching documents and generating answer..."):
            try:
                result = query_backend(
                    query=user_input,
                    session_id=st.session_state.session_id
                )
                
                # Handle different response formats
                if isinstance(result, dict):
                    answer = result.get("messages", [{}])[-1].get("content", "No answer generated")
                    sources = result.get("messages", [{}])[-1].get("sources", [])
                    confidence = result.get("binary_score")
                    route = result.get("route")
                else:
                    answer = str(result)
                    sources = []
                    confidence = None
                    route = None
                
                # Display answer
                st.write(answer)
                
                # Store in history with metadata
                message_data = {
                    "role": "assistant",
                    "content": answer,
                }
                
                if sources:
                    message_data["sources"] = sources
                    with st.expander("📚 Sources", expanded=False):
                        for source in sources:
                            st.info(f"📄 {source}")
                
                if confidence or route:
                    message_data["metadata"] = {}
                    if confidence:
                        message_data["metadata"]["confidence"] = 0.95  # Default
                        st.metric("Match Quality", f"{confidence}")
                    if route:
                        st.caption(f"Query routed to: {route}")
                
                st.session_state.messages.append(message_data)
                
            except Exception as e:
                st.error(f"⚠️ Error querying backend: {str(e)}")
                st.session_state.messages.pop()  # Remove failed user message

# Footer
st.divider()
st.caption("💡 Tip: Be specific in your questions for better results. Example: 'What are the key technical skills mentioned in the documents?'")
st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")