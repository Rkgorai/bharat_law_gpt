import streamlit as st
import sys
import os

import warnings

# Suppress annoying logging and warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", module="langgraph")

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# We import the classes, BUT we don't instantiate them yet
from src.rag.search import RAGSearch
from src.voice.voice_handler import VoiceHandler
from src.agent.agent import create_legal_agent

@st.cache_resource(show_spinner=False)
def get_rag_engine(db_path, model_name):
    """
    Initializes the RAG engine only once. 
    If you switch pages, Streamlit instantly returns this cached object.
    """
    if os.environ.get("BHARAT_LAW_VERBOSE") == "1":
        print(f"[INFO] Loading RAG Engine: {model_name}")
    return RAGSearch(persist_dir=db_path, llm_model=model_name)

@st.cache_resource(show_spinner=False)
def get_legal_agent(model_name):
    """
    Initializes the LangChain Legal Agent only once.
    """
    if os.environ.get("BHARAT_LAW_VERBOSE") == "1":
        print(f"[INFO] Loading Legal Agent: {model_name}")
    return create_legal_agent(llm_model=model_name)

@st.cache_resource(show_spinner=False)
def get_voice_handler():
    """
    Initializes the Voice Handler only once.
    """
    if os.environ.get("BHARAT_LAW_VERBOSE") == "1":
        print("[INFO] Loading Voice Handler...")
    return VoiceHandler()