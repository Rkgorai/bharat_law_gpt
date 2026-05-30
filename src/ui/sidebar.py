import streamlit as st

from src.ui.chat_handler import clear_chat_and_recordings

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ About")
        st.markdown(
            "**Bharat Law GPT** is an AI-powered legal assistant tailored for Indian law queries, providing accurate analysis and automated drafting.\n\n"
            "Click the ▶ play button next to any assistant response to listen to it."
        )
        st.markdown("---")
        if st.button("🗑️ Clear Chat", help="Wipe out all messages and local recordings from disk"):
            clear_chat_and_recordings()
            st.rerun()
