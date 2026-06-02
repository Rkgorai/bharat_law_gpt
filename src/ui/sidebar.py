import streamlit as st

from src.ui.chat_handler import clear_chat_and_recordings, AVAILABLE_MODELS

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        try:
            current_index = list(AVAILABLE_MODELS.values()).index(st.session_state.current_model)
        except ValueError:
            current_index = 0
            
        selected_label = st.selectbox(
            "Select Legal Model", 
            options=list(AVAILABLE_MODELS.keys()), 
            index=current_index
        )
        if AVAILABLE_MODELS[selected_label] != st.session_state.current_model:
            st.session_state.current_model = AVAILABLE_MODELS[selected_label]
            st.session_state.agent_system = None
            st.rerun()
            
        # Theme Selector
        theme_options = {
            "🌐 System Theme": "system",
            "☀️ Light Mode": "light",
            "🌙 Dark Mode": "dark"
        }
        
        try:
            current_theme_index = list(theme_options.values()).index(st.session_state.get("theme", "system"))
        except ValueError:
            current_theme_index = 0
            
        selected_theme_label = st.selectbox(
            "Select Theme",
            options=list(theme_options.keys()),
            index=current_theme_index
        )
        if theme_options[selected_theme_label] != st.session_state.get("theme"):
            st.session_state.theme = theme_options[selected_theme_label]
            st.rerun()
            
        st.markdown("---")
        st.markdown("### 🎛️ Hybrid Search Balance")
        bm25_w = st.slider(
            "BM25 (Keywords) Weight",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("bm25_weight", 0.6),
            step=0.05,
            help="Higher values favor exact legal keyword matches. Lower values favor conceptual similarity."
        )
        vector_w = round(1.0 - bm25_w, 2)
        st.write(f"Vector (Semantic) Weight: **{vector_w}**")
        st.session_state.bm25_weight = bm25_w
        st.session_state.vector_weight = vector_w
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            "**Bharat Law GPT** is an AI-powered legal assistant tailored for Indian law queries, providing accurate analysis and automated drafting.\n\n"
            "Click the ▶ play button next to any assistant response to listen to it."
        )
        st.markdown("---")
        if st.button("🗑️ Clear Chat", help="Wipe out all messages and local recordings from disk", use_container_width=True):
            clear_chat_and_recordings()
            st.rerun()
