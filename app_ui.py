import streamlit as st
import os
import sys

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Import modular components
from src.ui.chat_handler import (
    initialize_session_state,
    toggle_theme,
    submit_text_query,
    render_chat_history,
    render_draft_editor,
    handle_message_playback,
    handle_autoplay_audio,
    handle_pending_query,
    AVAILABLE_MODELS
)
from src.ui.styles import inject_custom_styles
from src.ui.voice_dictation import render_dictation_mic
from src.ui.send_button import render_send_button

# --- CONFIGURATION ---
PAGE_TITLE = "Bharat Law GPT"
PAGE_ICON = "⚖️"

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# --- INITIALIZE SESSION STATE ---
initialize_session_state()

# --- ADAPTIVE CSS & THEME INJECTION ---
inject_custom_styles()

# --- CUSTOM NAVBAR ---
nav_col1, nav_col2 = st.columns([12, 1])
with nav_col1:
    st.markdown('<div class="gemini-header">Bharat Law GPT</div>', unsafe_allow_html=True)
with nav_col2:
    btn_icon = "🌙" if st.session_state.theme == "light" else "☀️"
    st.button(btn_icon, on_click=toggle_theme, help="Toggle Light/Dark Mode")

st.markdown("---")

# --- MAIN UI ---
# 1. Render Chat History
render_chat_history()

# 2. Render Draft Editor
render_draft_editor()

# 3. Audio Auto-Play
handle_message_playback()
handle_autoplay_audio()

# 4. Handle Pending Query (from Text Input Enter / Dictation)
handle_pending_query()

# 5. Gemini-Style Big Oval Chatbar
st.markdown('<span id="chatbar-marker"></span>', unsafe_allow_html=True)
with st.container():
    col1, col2, col3, col4 = st.columns([10, 3, 1, 1])
    
    with col3:
        render_dictation_mic()
    
    with col1:
        st.text_input(
            "Ask something...", 
            key="chat_input_box", 
            label_visibility="collapsed",
            placeholder="Ask a legal question...",
            on_change=submit_text_query
        )

    with col2:
        try:
            current_index = list(AVAILABLE_MODELS.values()).index(st.session_state.current_model)
        except ValueError:
            current_index = 0
            
        selected_label = st.selectbox(
            "Model", 
            options=list(AVAILABLE_MODELS.keys()), 
            index=current_index, 
            label_visibility="collapsed"
        )
        if AVAILABLE_MODELS[selected_label] != st.session_state.current_model:
            st.session_state.current_model = AVAILABLE_MODELS[selected_label]
            st.session_state.agent_system = None
            st.rerun()

    with col4:
        render_send_button()

# Hot reload trigger to clear Streamlit module cache and load new mobile styles (v9)