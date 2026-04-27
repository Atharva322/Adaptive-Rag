"""
Main chat interface with RAG capabilities and source document display.
"""

import streamlit as st
import requests
import json
import uuid
import os
from datetime import datetime
import sys
sys.path.append("..")
from utils.api_client import (
    query_backend,
    document_upload_rag,
    BASE_API_URL,
    delete_document,
    evaluate_ragas,
)


st.set_page_config(
    page_title="Adaptive RAG - Chat",
    page_icon="💬",
    layout="wide"
)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "last_uploaded_doc" not in st.session_state:
    st.session_state.last_uploaded_doc = None
if "chat_loaded" not in st.session_state:
    try:
        resp = requests.get(
            f"{BASE_API_URL}/rag/chat_history",
            params={"session_id": st.session_state.session_id}
        )
        if resp.status_code == 200:
            history = resp.json().get("messages", [])
            st.session_state.messages = history
        st.session_state.chat_loaded = True
    except:
        st.session_state.chat_loaded = True
if "enable_ragas_eval" not in st.session_state:
    st.session_state.enable_ragas_eval = False
if "ragas_ground_truth" not in st.session_state:
    st.session_state.ragas_ground_truth = ""
if "ragas_selected_metrics" not in st.session_state:
    st.session_state.ragas_selected_metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

# Check authentication
_disable_auth = os.getenv("DISABLE_AUTH", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
if "jwt_token" not in st.session_state or st.session_state.jwt_token is None:
    if _disable_auth:
        st.session_state.jwt_token = "local_dev_token"
        st.session_state.username = st.session_state.get("username") or "local_dev"
    else:
        st.warning("Please log in first")
        st.stop()

# Header
col1, col2 = st.columns([0.9, 0.1])
with col1:
    st.title("Adaptive RAG Chat")
    st.markdown("Ask questions about your uploaded documents")

with col2:
    if st.button("Logout"):
        st.session_state.jwt_token = None
        st.session_state.session_id = None
        st.rerun()

# Sidebar - Document Management
with st.sidebar:
    st.header("Document Management")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "docx", "md"],
        help="Upload a document to query against"
    )

    # Refresh documents
    if st.button("Refresh Documents", use_container_width=True):
        try:
            resp = requests.get(f"{BASE_API_URL}/rag/documents")
            if resp.status_code == 200:
                st.session_state.uploaded_files = resp.json().get("documents", [])
                st.success("Documents refreshed!")
            else:
                st.error("Failed to load documents")
        except Exception as e:
            st.error(f"Error: {str(e)}")
        st.rerun()

    if uploaded_file:
        doc_description = st.text_area(
            "Document Description",
            placeholder="Briefly describe what this document contains...",
            height=80
        )

        if st.button("Upload Document", use_container_width=True):
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
                            st.session_state.last_uploaded_doc = {
                                "name": uploaded_file.name,
                                "description": doc_description
                            }
                            st.success(f"{uploaded_file.name} uploaded successfully!")
                            st.rerun()
                        else:
                            st.error(f"Upload failed: {result.get('message', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Error uploading document: {str(e)}")

    # Uploaded files list
    if st.session_state.uploaded_files:
        st.divider()
        st.subheader("Uploaded Documents")
        for idx, file_info in enumerate(st.session_state.uploaded_files):
            with st.expander(f"{file_info['name']}"):
                st.write(f"**Description:** {file_info['description']}")
                st.write(f"**Uploaded:** {file_info['uploaded_at']}")

                if st.button("Delete", key=f"delete_{idx}", use_container_width=True):
                    result = delete_document(file_info["name"])
                    if result.get("status") == "success":
                        st.session_state.uploaded_files.pop(idx)
                        st.success(f"'{file_info['name']}' deleted")
                        st.rerun()
                    else:
                        st.error(f"Delete failed: {result.get('message', 'Unknown error')}")

    st.divider()
    st.subheader("RAGAS Evaluation")
    st.session_state.enable_ragas_eval = st.toggle(
        "Show RAGAS scores",
        value=st.session_state.enable_ragas_eval,
        help="Optional: evaluate each response with RAGAS using your ground truth.",
    )
    if st.session_state.enable_ragas_eval:
        st.session_state.ragas_ground_truth = st.text_area(
            "Ground truth for next question",
            value=st.session_state.ragas_ground_truth,
            placeholder="Write the expected ideal answer for the next question...",
            height=100,
            help="RAGAS uses this as reference to score faithfulness/relevancy/recall/precision.",
        )
        st.session_state.ragas_selected_metrics = st.multiselect(
            "Metrics",
            options=[
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ],
            default=st.session_state.ragas_selected_metrics,
        )

# Main chat area
st.subheader("Conversation")

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])

            if "sources" in message and message["sources"]:
                with st.expander("Sources", expanded=False):
                    cols = st.columns(len(message["sources"]))
                    for idx, source in enumerate(message["sources"]):
                        with cols[idx]:
                            st.info(f"{source}")

            if "metadata" in message:
                meta = message["metadata"]
                col1, col2 = st.columns(2)
                with col1:
                    if "confidence" in meta:
                        st.metric("Confidence", f"{meta['confidence']*100:.0f}%")
                with col2:
                    if "processing_time" in meta:
                        st.metric("Processing Time", f"{meta['processing_time']:.2f}s")
            if "ragas_scores" in message:
                ragas_data = message["ragas_scores"]
                with st.expander("RAGAS Scores", expanded=False):
                    aggregate = ragas_data.get("aggregate_scores", {})
                    if aggregate:
                        metric_cols = st.columns(len(aggregate))
                        for idx, (metric_name, metric_value) in enumerate(aggregate.items()):
                            with metric_cols[idx]:
                                if isinstance(metric_value, float):
                                    st.metric(metric_name, f"{metric_value:.3f}")
                                else:
                                    st.metric(metric_name, str(metric_value))
                    else:
                        st.caption("No aggregate RAGAS scores available for this message.")

# Chat input
st.divider()
user_input = st.chat_input(
    "Ask a question about your documents...",
    key="chat_input"
)

if user_input:
    contextual_keywords = ["this document", "this paper", "the document", "the paper", "it"]
    query_lower = user_input.lower()

    if any(keyword in query_lower for keyword in contextual_keywords) and st.session_state.last_uploaded_doc:
        enhanced_query = f"{user_input}\n\nContext: Referring to the document '{st.session_state.last_uploaded_doc['name']}' which is about: {st.session_state.last_uploaded_doc['description']}"
    else:
        enhanced_query = user_input

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                result = query_backend(
                    query=enhanced_query,
                    session_id=st.session_state.session_id
                )

                if isinstance(result, dict):
                    answer = result.get("messages", [{}])[-1].get("content", "No answer generated")
                    sources = result.get("messages", [{}])[-1].get("sources", [])
                    retrieved_contexts = result.get("retrieved_contexts", [])
                    confidence = result.get("binary_score")
                    route = result.get("route")
                else:
                    answer = str(result)
                    sources = []
                    retrieved_contexts = []
                    confidence = None
                    route = None

                st.write(answer)

                message_data = {
                    "role": "assistant",
                    "content": answer,
                }

                if sources:
                    message_data["sources"] = sources
                    with st.expander("Sources", expanded=False):
                        for source in sources:
                            st.info(f"{source}")

                if confidence or route:
                    message_data["metadata"] = {}
                    if confidence:
                        message_data["metadata"]["confidence"] = 0.95
                        st.metric("Match Quality", f"{confidence}")
                    if route:
                        st.caption(f"Query routed to: {route}")

                if st.session_state.enable_ragas_eval:
                    ground_truth = st.session_state.ragas_ground_truth.strip()
                    if ground_truth:
                        with st.spinner("Computing RAGAS scores..."):
                            ragas_result = evaluate_ragas(
                                question=enhanced_query,
                                ground_truth=ground_truth,
                                answer=answer,
                                contexts=retrieved_contexts,
                                include_per_sample=True,
                                metrics=st.session_state.ragas_selected_metrics,
                            )
                        if ragas_result.get("status") == "success":
                            ragas_data = ragas_result.get("data", {})
                            message_data["ragas_scores"] = ragas_data
                            with st.expander("RAGAS Scores", expanded=False):
                                aggregate = ragas_data.get("aggregate_scores", {})
                                if aggregate:
                                    metric_cols = st.columns(len(aggregate))
                                    for idx, (metric_name, metric_value) in enumerate(aggregate.items()):
                                        with metric_cols[idx]:
                                            if isinstance(metric_value, float):
                                                st.metric(metric_name, f"{metric_value:.3f}")
                                            else:
                                                st.metric(metric_name, str(metric_value))
                                else:
                                    st.caption("No aggregate scores returned by backend.")
                        else:
                            st.warning(f"RAGAS evaluation failed: {ragas_result.get('message', 'Unknown error')}")
                    else:
                        st.caption("RAGAS is enabled. Add a ground truth in the sidebar to score this answer.")

                st.session_state.messages.append(message_data)

            except Exception as e:
                st.error(f"Error querying backend: {str(e)}")
                st.session_state.messages.pop()

# Footer
st.divider()
st.caption("Tip: Be specific in your questions for better results.")
st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")
