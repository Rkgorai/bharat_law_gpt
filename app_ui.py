import streamlit as st
import os
import time
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
    "openai/gpt-oss-20b (Accurate)": "openai/gpt-oss-20b",
    "qwen/qwen3-32b-chat (Powerful)": "qwen/qwen3-32b"
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
        st.info("Run 'build_db.py' to create the database.")

    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- RAG INITIALIZATION (LAZY LOADING) ---

@st.cache_resource(show_spinner=False)
def get_rag_engine(persist_dir, model_name):
    """
    Cached loader. This keeps the heavy Embedding Model in memory 
    even if you refresh the page.
    """
    print(f"[INFO] Loading RAG Engine with {model_name}...") 
    return RAGSearch(persist_dir=persist_dir, llm_model=model_name)

def ensure_system_ready():
    """Lazy loader wrapper."""
    if st.session_state.rag_system is None:
        with st.spinner(f"⚡ Activating {st.session_state.current_model}..."):
            st.session_state.rag_system = get_rag_engine(DB_PATH, st.session_state.current_model)

# --- CHAT INTERFACE ---

# 1. Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Handle Input
if prompt := st.chat_input("Ask a legal question..."):
    # User
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Assistant
    with st.chat_message("assistant"):
        # A. Check Database
        if not os.path.exists(os.path.join(DB_PATH, "faiss.index")):
            st.error("Vector Database not found. Please run the build script.")
            st.stop()
            
        # B. Lazy Load System (If not ready)
        ensure_system_ready()
        
        # C. Generate Answer
        try:
            with st.spinner("Analyzing legal docs..."):
                # 1. Prepare History (Exclude current prompt to avoid duplication)
                history_for_llm = st.session_state.messages[:-1]
                
                # 2. Call Search with History
                result = st.session_state.rag_system.search_and_summarize(
                    query=prompt, 
                    chat_history=history_for_llm
                )
                
                # 3. Extract Data
                answer_text = result["answer"]
                sources = result["sources"]

                # 4. Display Answer
                st.markdown(answer_text)
                
                # 5. Display Sources in Expander
                if sources:
                    with st.expander("📚 View Legal Sources"):
                        for source in sources:
                            st.caption(f"📄 {source}")
                
                # 6. Save to History
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
                
        except Exception as e:
            st.error(f"Error generating response: {e}")