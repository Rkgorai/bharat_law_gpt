import streamlit as st
import os
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# --- CONFIGURATION ---
PAGE_TITLE = "Bharat Law GPT"
PAGE_ICON = "⚖️"
DB_PATH = "db/faiss_store"

# Supported Models on Groq
AVAILABLE_MODELS = {
    "Llama 3.1 8B (Fastest)": "llama-3.1-8b-instant",
    "Kimi K2 Instruct (Moonshot)": "moonshotai/kimi-k2-instruct-0905",
    "Mixtral 8x7B (Balanced)": "mixtral-8x7b-32768",
    "Gemma 2 9B (Google)": "gemma2-9b-it",
    "Llama 3 70B (Smartest)": "llama3-70b-8192"
}

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

# --- CUSTOM CSS FOR STICKY HEADER ---
st.markdown(
    """
    <style>
        /* Hide default main block padding to make header sit flush */
        .block-container {
            padding-top: 3rem;
        }
        
        /* Sticky Header Class */
        .sticky-header {
            position: sticky;
            top: 0;
            z-index: 999;
            background-color: rgba(255, 255, 255, 0.95); /* Translucent white */
            backdrop-filter: blur(10px);
            padding: 10px 0px;
            border-bottom: 1px solid rgba(49, 51, 63, 0.1);
            margin-bottom: 20px;
        }
        
        /* Dark mode adjustment */
        @media (prefers-color-scheme: dark) {
            .sticky-header {
                background-color: rgba(14, 17, 23, 0.95); /* Streamlit dark bg */
                border-bottom: 1px solid rgba(250, 250, 250, 0.1);
            }
        }
    </style>
    
    <div class="sticky-header">
        <h1 style="margin:0; padding:0;">⚖️ 🇮🇳 Bharat Law GPT</h1>
        <small style="color: gray;">Your AI Legal Assistant for Indian Constitution & Acts</small>
    </div>
    """,
    unsafe_allow_html=True
)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_system" not in st.session_state:
    st.session_state.rag_system = None

if "current_model" not in st.session_state:
    st.session_state.current_model = "llama-3.1-8b-instant"

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. Model Selection Dropdown
    selected_label = st.selectbox(
        "Select AI Model",
        options=list(AVAILABLE_MODELS.keys()),
        index=0
    )
    selected_model_id = AVAILABLE_MODELS[selected_label]

    # Detect Model Change
    if selected_model_id != st.session_state.current_model:
        st.session_state.current_model = selected_model_id
        st.session_state.rag_system = None # Force reload
        st.toast(f"Switched to {selected_label}", icon="🔄")

    st.divider()
    
    # 2. System Status
    index_path = os.path.join(DB_PATH, "faiss.index")
    if os.path.exists(index_path):
        st.success(f"✅ Database Active")
        st.caption(f"Model: `{selected_model_id}`")
    else:
        st.error("❌ Database Missing")
        st.info("Run backend build script.")

    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- RAG INITIALIZATION ---
def initialize_rag():
    """Initializes RAG with the currently selected model."""
    if st.session_state.rag_system is None:
        try:
            # Pass the dynamic model ID to the RAG class
            st.session_state.rag_system = RAGSearch(
                persist_dir=DB_PATH, 
                llm_model=st.session_state.current_model
            )
        except Exception as e:
            st.error(f"Failed to load model {st.session_state.current_model}: {e}")

# 1. Load System
if os.path.exists(os.path.join(DB_PATH, "faiss.index")):
    if st.session_state.rag_system is None:
        with st.spinner(f"Initializing {selected_label}..."):
            initialize_rag()
else:
    st.warning("Please ensure the Vector Database is built.")
    st.stop()

# --- CHAT INTERFACE ---

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle Input
if prompt := st.chat_input("Ask a legal question..."):
    # User
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Assistant
    if st.session_state.rag_system:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing legal docs..."):
                try:
                    response = st.session_state.rag_system.search_and_summarize(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {e}")
    else:
        st.error("System not initialized.")