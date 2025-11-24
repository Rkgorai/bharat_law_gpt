import streamlit as st
import os
import time
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# --- CONFIGURATION ---
PAGE_TITLE = "Bharat Law GPT"
PAGE_ICON = "⚖️"
DB_PATH = "db/faiss_store"
DATA_DIR = "legal_docs"

# Supported Models
AVAILABLE_MODELS = {
    "Llama 3.1 8B (Fastest)": "llama-3.1-8b-instant",
    "Kimi K2 Instruct (Moonshot)": "moonshotai/kimi-k2-instruct-0905",
    "openai/gpt-oss-20b (Accurate)": "openai/gpt-oss-20b",
    "qwen/qwen3-32b-chat (Powerful)": "qwen/qwen3-32b"
}

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

# --- CUSTOM CSS ---
st.markdown(
    """
    <style>
        .block-container { padding-top: 3rem; }
        .sticky-header {
            position: sticky;
            top: 0;
            z-index: 999;
            background-color: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 15px 0px;
            border-bottom: 1px solid rgba(49, 51, 63, 0.1);
            margin-bottom: 20px;
        }
        @media (prefers-color-scheme: dark) {
            .sticky-header {
                background-color: rgba(14, 17, 23, 0.95);
                border-bottom: 1px solid rgba(250, 250, 250, 0.1);
            }
        }
    </style>
    <div class="sticky-header">
        <h1 style="margin:0; padding:0; font-size: 2.2rem;">⚖️ 🇮🇳 Bharat Law GPT</h1>
        <small style="color: gray; font-size: 0.9rem;">Your AI Legal Assistant for Indian Constitution & Acts</small>
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

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    selected_label = st.selectbox("Select Brain", options=list(AVAILABLE_MODELS.keys()), index=0)
    selected_model_id = AVAILABLE_MODELS[selected_label]

    if selected_model_id != st.session_state.current_model:
        st.session_state.current_model = selected_model_id
        st.session_state.rag_system = None
        st.toast(f"Brain switched to: {selected_label}", icon="🧠")

    st.divider()
    if os.path.exists(os.path.join(DB_PATH, "faiss.index")):
        st.success("✅ Knowledge Base Active")
    else:
        st.error("❌ Database Missing")
        st.info("Run 'build_db.py'")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()

# --- LAZY LOAD RAG ---
@st.cache_resource(show_spinner=False)
def get_rag_engine(persist_dir, model_name):
    return RAGSearch(persist_dir=persist_dir, llm_model=model_name)

def ensure_system_ready():
    if st.session_state.rag_system is None:
        with st.spinner(f"⚡ Activating {st.session_state.current_model}..."):
            st.session_state.rag_system = get_rag_engine(DB_PATH, st.session_state.current_model)

# --- CHAT UI ---
# 1. Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Sources Area (Dropdown for LATEST response only)
if st.session_state.last_sources:
    with st.expander("📚 View References"):
        for source in st.session_state.last_sources:
            try:
                # Source format from DB: "C:\Path\To\File.pdf (Pg 5)"
                file_part, page_part = source.rsplit(" (Pg ", 1)
                clean_filename = os.path.basename(file_part)
                page_num_str = page_part.replace(")", "")
                st.markdown(f"- 📄 **{clean_filename}** (Page {page_num_str})")
            except Exception:
                st.markdown(f"- {source}")

# 3. Handle Input
if prompt := st.chat_input("Ask a legal question..."):
    # User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Assistant Response
    with st.chat_message("assistant"):
        if not os.path.exists(os.path.join(DB_PATH, "faiss.index")):
            st.error("Vector Database not found. Please run the build script.")
            st.stop()
            
        ensure_system_ready()
        
        try:
            with st.spinner("Analyzing legal docs..."):
                history_for_llm = st.session_state.messages[:-1]
                
                result = st.session_state.rag_system.search_and_summarize(
                    query=prompt, 
                    chat_history=history_for_llm
                )
                
                answer_text = result["answer"]
                raw_sources = result["sources"]

                # --- LOGIC FIX: HIDE SOURCES IF ANSWER IS NEGATIVE ---
                answer_lower = answer_text.lower()
                negative_phrases = [
                    "i could not find", 
                    "no relevant documents", 
                    "i do not have any information", 
                    "i don't have any information",
                    "does not contain information",
                    "i cannot cite"
                ]
                
                if any(phrase in answer_lower for phrase in negative_phrases):
                    raw_sources = []

                # Update Session State
                st.session_state.last_sources = raw_sources
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
                
                # Display Answer
                st.markdown(answer_text)
                
                # Rerun to show the updated sources dropdown (or hide it)
                st.rerun()
                
        except Exception as e:
            st.error(f"Error generating response: {e}")