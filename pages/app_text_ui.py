import streamlit as st
import os
import sys

# --- PATH SETUP ---
# Necessary so we can import 'src' from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ui.dependencies import get_legal_agent
from langchain_core.messages import HumanMessage, AIMessage
import re
from src.ui.pdf_utils import create_pdf

# --- CONFIGURATION ---
PAGE_TITLE = "Bharat Law - Text Chat"
PAGE_ICON = "💬"
DB_PATH = "db/faiss_store"

# Supported Models
AVAILABLE_MODELS = {
    "Llama 4 Scout 17B (Instruct)": "meta-llama/llama-4-scout-17b-16e-instruct",
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
if "agent_system" not in st.session_state:
    st.session_state.agent_system = None
if "current_model" not in st.session_state:
    st.session_state.current_model = "meta-llama/llama-4-scout-17b-16e-instruct" # Default model
if "draft_content" not in st.session_state:
    st.session_state.draft_content = None

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
        st.session_state.agent_system = None 
        st.toast(f"Brain switched to: {selected_label}", icon="🧠")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- LOAD ENGINE ---
def ensure_system_ready():
    # Uses shared.py cache
    if st.session_state.agent_system is None:
        with st.spinner(f"⚡ Accessing {st.session_state.current_model}..."):
            st.session_state.agent_system = get_legal_agent(st.session_state.current_model)

# --- CHAT UI ---
st.title("💬 Bharat Law Legal Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.draft_content:
    st.divider()
    st.subheader("📝 Draft Editor")
    st.info("You can edit the text directly in the box below before downloading.")
    edited_draft = st.text_area("Review and edit your document:", value=st.session_state.draft_content, height=400)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("💾 Save Changes to Draft"):
            st.session_state.draft_content = edited_draft
            st.success("Draft updated!")
    with col2:
        try:
            pdf_bytes = create_pdf(edited_draft)
            st.download_button(
                label="📄 Download as PDF",
                data=pdf_bytes,
                file_name="BharatLaw_Draft.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
    st.divider()

if prompt := st.chat_input("Ask a legal question..."):
    # User Input
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Assistant Response
    with st.chat_message("assistant"):
        ensure_system_ready()
        try:
            history_for_llm = st.session_state.messages[:-1]
            
            formatted_history = []
            for msg in history_for_llm:
                if msg["role"] == "user":
                    formatted_history.append(HumanMessage(content=msg["content"]))
                else:
                    formatted_history.append(AIMessage(content=msg["content"]))
            
            formatted_history.append(HumanMessage(content=prompt))
            
            status_container = st.status("Agent is thinking...", expanded=True)
            
            answer_text = ""
            # Stream events from LangGraph
            for event in st.session_state.agent_system.stream({"messages": formatted_history}):
                for node_name, value in event.items():
                    if "messages" in value:
                        messages = value["messages"]
                        if not isinstance(messages, list):
                            messages = [messages]
                        for msg in messages:
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    status_container.write(f"🛠️ Calling `{tool_call['name']}`...")
                            elif getattr(msg, "type", "") == "tool":
                                status_container.write(f"✅ Finished `{msg.name}`")
                                if "<DRAFT_SAVED>" in str(msg.content):
                                    # Read the file directly instead of waiting for the LLM
                                    if os.path.exists("db/latest_draft.txt"):
                                        with open("db/latest_draft.txt", "r") as f:
                                            st.session_state.draft_content = f.read()
                                        st.session_state.messages.append({"role": "assistant", "content": "\n\n📝 **I have generated the draft for you.** You can review, edit, and download it in the Draft Editor above.\n\n"})
                                        status_container.update(label="Draft Complete!", state="complete", expanded=False)
                                        st.rerun()
                            
                            if getattr(msg, "type", "") == "ai" and msg.content:
                                answer_text += msg.content
                                
            status_container.update(label="Finished!", state="complete", expanded=False)
            
            # Normal chat logic if no draft was intercepted
            if answer_text:
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
                st.markdown(answer_text)
                
        except Exception as e:
            # If the LLM still crashes for some unrelated reason, just show the error
            st.error(f"Error: {e}")