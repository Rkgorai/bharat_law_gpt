import streamlit as st
import os
import sys

# --- PATH SETUP ---
# Necessary so we can import 'src' from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.shared import get_rag_engine

# --- CONFIGURATION ---
PAGE_TITLE = "Bharat Law - Text Chat"
PAGE_ICON = "💬"
DB_PATH = "db/faiss_store"

# Supported Models
AVAILABLE_MODELS = {
    "Kimi K2 Instruct (Moonshot)": "moonshotai/kimi-k2-instruct-0905",
    "Llama 3.1 8B (Fastest)": "llama-3.1-8b-instant",
    "openai/gpt-oss-20b (Accurate)": "openai/gpt-oss-20b",
    "qwen/qwen3-32b-chat (Powerful)": "qwen/qwen3-32b"
}

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

# --- CSS ---
st.markdown("""<style>.block-container { padding-top: 2rem; }</style>""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "current_model" not in st.session_state:
    st.session_state.current_model = "llama-3.1-8b-instant"

# --- SIDEBAR ---
with st.sidebar:
    # Navigation Buttons
    if st.button("🏠 Back to Home"):
        st.switch_page("app_ui.py")
        
    st.header("⚙️ Settings")
    
    if st.button("🎙️ Switch to Voice"):
        st.switch_page("pages/app_voice_ui.py")
        
    st.divider()

    # Brain Selection
    selected_label = st.selectbox("Select Brain", options=list(AVAILABLE_MODELS.keys()), index=0)
    selected_model_id = AVAILABLE_MODELS[selected_label]

    if selected_model_id != st.session_state.current_model:
        st.session_state.current_model = selected_model_id
        # Reset reference to force update
        st.session_state.rag_system = None 
        st.toast(f"Brain switched to: {selected_label}", icon="🧠")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- LOAD ENGINE ---
def ensure_system_ready():
    # Uses shared.py cache
    if st.session_state.rag_system is None:
        with st.spinner(f"⚡ Accessing {st.session_state.current_model}..."):
            st.session_state.rag_system = get_rag_engine(DB_PATH, st.session_state.current_model)

# --- CHAT UI ---
st.title("💬 Bharat Law Legal Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a legal question..."):
    # User Input
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Assistant Response
    with st.chat_message("assistant"):
        ensure_system_ready()
        try:
            with st.spinner("Analyzing..."):
                history_for_llm = st.session_state.messages[:-1]
                
                result = st.session_state.rag_system.search_and_summarize(
                    query=prompt, 
                    chat_history=history_for_llm
                )
                
                answer_text = result["answer"]
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
                st.markdown(answer_text)
        except Exception as e:
            st.error(f"Error: {e}")