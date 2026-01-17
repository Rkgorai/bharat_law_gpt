import streamlit as st
import sys
import os

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# We import the classes, BUT we don't instantiate them yet
from src.search import RAGSearch
from src.voice_handler import VoiceHandler

@st.cache_resource(show_spinner=False)
def get_rag_engine(db_path, model_name):
    """
    Initializes the RAG engine only once. 
    If you switch pages, Streamlit instantly returns this cached object.
    """
    print(f"[INFO] Loading RAG Engine: {model_name}")
    return RAGSearch(persist_dir=db_path, llm_model=model_name)

@st.cache_resource(show_spinner=False)
def get_voice_handler():
    """
    Initializes the Voice Handler only once.
    """
    print("[INFO] Loading Voice Handler...")
    return VoiceHandler()