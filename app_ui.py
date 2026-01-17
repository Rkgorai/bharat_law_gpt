import streamlit as st

# --- CONFIGURATION ---
PAGE_TITLE = "Bharat Law GPT"
PAGE_ICON = "⚖️"

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

# --- CUSTOM CSS ---
st.markdown(
    """
    <style>
        .main-header {
            font-size: 3rem;
            text-align: center;
            color: #FF9933; /* Saffron */
            margin-bottom: 10px;
            font-weight: bold;
        }
        .sub-header {
            font-size: 1.2rem;
            text-align: center;
            color: #555;
            margin-bottom: 40px;
        }
        /* Make buttons look like cards */
        div.stButton > button {
            height: 120px;
            font-size: 22px;
            border-radius: 15px;
            width: 100%;
            border: 2px solid #f0f0f0;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            border-color: #FF9933;
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- UI ---
st.markdown("<div class='main-header'>⚖️ Bharat Law GPT</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Your AI Companion for Indian Legal Knowledge</div>", unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.info("💬 **Text Mode**\n\nClassic chat interface. Best for detailed queries, reading long answers, and copying text.")
    if st.button("Enter Text Chat ➜"):
        st.switch_page("pages/app_text_ui.py")

with col2:
    st.success("🎙️ **Voice Mode**\n\nHands-free interaction. Best for quick questions and listening to answers on the go.")
    if st.button("Enter Voice Chat ➜"):
        st.switch_page("pages/app_voice_ui.py")

st.divider()
st.caption("© 2024 Bharat Law AI | Powered by RAG & LLMs")