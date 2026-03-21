import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Adaptive RAG", layout="wide")

# Skip authentication - go straight to chat
st.session_state.authenticated = True
st.session_state.jwt_token = "dummy_token"

# Title and navigation
st.title("🤖 Adaptive RAG Chat")
st.write("Upload documents and ask questions about them.")

# Redirect to chat page
st.switch_page("pages/01_chat.py")