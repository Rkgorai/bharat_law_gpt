import streamlit as st
import os
import sys
import base64
import time

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.ui.dependencies import get_legal_agent, get_voice_handler
from src.ui.pdf_utils import create_pdf
from langchain_core.messages import HumanMessage, AIMessage
from streamlit_mic_recorder import speech_to_text
from mutagen.mp3 import MP3

# --- CONFIGURATION ---
PAGE_TITLE = "Bharat Law GPT"
PAGE_ICON = "⚖️"

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# --- SESSION STATE ---
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_system" not in st.session_state:
    st.session_state.agent_system = None
if "current_model" not in st.session_state:
    st.session_state.current_model = "meta-llama/llama-4-scout-17b-16e-instruct"
if "draft_content" not in st.session_state:
    st.session_state.draft_content = None
if "audio_to_play" not in st.session_state:
    st.session_state.audio_to_play = None
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# --- ADAPTIVE CSS & THEME INJECTION ---
if st.session_state.theme == "dark":
    theme_css = """
    :root {
        --bg-color: #131314;
        --text-color: #e3e3e3;
        --chat-user: #282a2c;
        --chat-bot: #1e1f20;
        --border-color: #3c4043;
        --shadow: 0 4px 15px rgba(0,0,0,0.5);
        --gradient-start: #000000;
        --gradient-end: #131314;
    }
    """
else:
    theme_css = """
    :root {
        --bg-color: #ffffff;
        --text-color: #202124;
        --chat-user: #e3f2fd;
        --chat-bot: #ffffff;
        --border-color: #dadce0;
        --shadow: 0 4px 15px rgba(0,0,0,0.1);
        --gradient-start: #e8f0fe;
        --gradient-end: #ffffff;
    }
    """

st.markdown(f"""
<style>
{theme_css}

/* 1. Transparent Default Header */
[data-testid="stHeader"] {{ 
    background-color: transparent !important; 
}}

/* 2. Base App Styling */
.stApp {{
    background: linear-gradient(135deg, var(--gradient-start) 0%, var(--gradient-end) 100%);
    color: var(--text-color) !important;
}}
.stApp * {{
    color: var(--text-color);
}}

/* 3. Layout Width */
.block-container {{
    padding-top: 1rem !important;
    padding-bottom: 150px !important;
    max-width: 1200px !important;
}}

/* 4. Chat Bubbles */
[data-testid="stChatMessage"] {{
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
    background-color: var(--chat-bot);
}}
[data-testid="stChatMessage"] * {{
    color: var(--text-color) !important;
}}
[data-testid="chat-message-user"] {{
    background-color: var(--chat-user) !important;
}}

/* 5. Tool Expanders */
[data-testid="stStatusWidget"] {{
    background-color: transparent !important;
    border: 1px dashed var(--border-color) !important;
    border-radius: 8px;
    margin-top: 10px;
}}
[data-testid="stStatusWidget"] * {{
    color: var(--text-color) !important;
}}

/* 6. GUARANTEED FIXED OVAL CHATBAR USING :has() */
div[data-testid="stHorizontalBlock"]:has(input[aria-label="Ask something..."]) {{
    position: fixed !important;
    bottom: 25px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: 90% !important;
    max-width: 1000px !important;
    background-color: var(--bg-color) !important;
    border: 2px solid var(--border-color) !important;
    border-radius: 40px !important;
    padding: 5px 25px !important;
    box-shadow: var(--shadow) !important;
    z-index: 9999 !important;
    align-items: center !important;
}}
div[data-testid="stHorizontalBlock"]:has(input[aria-label="Ask something..."]):focus-within {{
    border-color: #4285F4 !important;
    box-shadow: 0 0 0 1px #4285F4 !important;
}}

/* 7. Transparent Inner Inputs */
.stTextInput div[data-baseweb="base-input"], .stTextInput div[data-baseweb="input"] {{
    background-color: transparent !important;
    border: none !important;
}}
.stTextInput input {{
    border: none !important;
    background-color: transparent !important;
    box-shadow: none !important;
    color: var(--text-color) !important;
    -webkit-text-fill-color: var(--text-color) !important;
    font-size: 16px !important;
    padding: 15px 5px !important; /* Taller input */
}}
.stTextInput input:focus {{
    background-color: transparent !important;
    box-shadow: none !important;
}}

/* 8. Model Selector styling (No border, hover effect) */
.stSelectbox > div > div {{
    border: none !important;
    background-color: transparent !important;
    box-shadow: none !important;
    color: #4285F4 !important;
    font-weight: 500;
    border-radius: 20px;
    transition: background-color 0.2s ease;
    cursor: pointer;
    margin-top: 5px !important;
}}
.stSelectbox > div > div:hover {{
    background-color: rgba(128, 128, 128, 0.1) !important;
}}
.stSelectbox label {{ display: none !important; }}

/* 9. Custom Top Nav */
div[data-testid="stHorizontalBlock"]:has(.gemini-header) {{
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    justify-content: space-between !important;
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    gap: 0 !important;
}}
div[data-testid="stHorizontalBlock"]:has(.gemini-header) > div[data-testid="column"]:nth-child(1) {{
    width: calc(100% - 60px) !important;
    flex: 0 0 calc(100% - 60px) !important;
    min-width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}}
div[data-testid="stHorizontalBlock"]:has(.gemini-header) > div[data-testid="column"]:nth-child(2) {{
    width: 60px !important;
    flex: 0 0 60px !important;
    min-width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}}
.gemini-header {{
    font-size: clamp(1.5rem, 5vw, 2.2rem);
    font-weight: 600;
    background: -webkit-linear-gradient(45deg, #4285F4, #EA4335, #FBBC05, #34A853);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<audio controls autoplay style="display:none"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'

def get_audio_duration(file_path):
    try:
        return MP3(file_path).info.length
    except Exception:
        return 5

def ensure_system_ready():
    if st.session_state.agent_system is None:
        with st.spinner(f"⚡ Booting {st.session_state.current_model}..."):
            st.session_state.agent_system = get_legal_agent(st.session_state.current_model)

def submit_text_query():
    query = st.session_state.chat_input_box
    if query.strip():
        st.session_state.pending_query = query
        st.session_state.chat_input_box = "" # Clear input

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

def process_query(query: str, use_voice: bool = False):
    st.session_state.messages.append({"role": "user", "content": query})
    ensure_system_ready()
    
    with st.chat_message("user"):
        st.markdown(query)
        
    with st.chat_message("assistant"):
        history_for_llm = st.session_state.messages[:-1]
        formatted_history = [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) 
            for m in history_for_llm
        ]
        formatted_history.append(HumanMessage(content=query))
        
        status_container = st.status("Thinking...", expanded=True)
        answer_text = ""
        
        try:
            for event in st.session_state.agent_system.stream({"messages": formatted_history}):
                for node_name, value in event.items():
                    if "messages" in value:
                        messages = value["messages"]
                        if not isinstance(messages, list):
                            messages = [messages]
                        for msg in messages:
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    status_container.write(f"⚙️ Running `{tool_call['name']}`")
                            elif getattr(msg, "type", "") == "tool":
                                status_container.write(f"✅ Extracted data from `{msg.name}`")
                                if "<DRAFT_SAVED>" in str(msg.content):
                                    if os.path.exists("db/latest_draft.txt"):
                                        with open("db/latest_draft.txt", "r") as f:
                                            st.session_state.draft_content = f.read()
                                        st.session_state.messages.append({"role": "assistant", "content": "I have generated the legal draft. Please review and export it in the editor below."})
                                        status_container.update(label="Draft Generated", state="complete", expanded=False)
                                        st.rerun()
                            
                            if getattr(msg, "type", "") == "ai" and msg.content:
                                answer_text += msg.content
                                
            status_container.update(label="Complete", state="complete", expanded=False)
            
            if answer_text:
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
                st.markdown(answer_text)
                
                if use_voice:
                    voice_handler = get_voice_handler()
                    audio_path = voice_handler.synthesize(answer_text)
                    if audio_path:
                        st.session_state.audio_to_play = audio_path
                        st.rerun()
                        
        except Exception as e:
            status_container.update(label="Error Occurred", state="error", expanded=True)
            st.error(f"Error: {e}")

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
if not st.session_state.messages:
    st.info("👋 Welcome! Type or speak a legal question to begin.")
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Render Draft Editor
if st.session_state.draft_content:
    st.markdown("---")
    edited_draft = st.text_area("Interactive Draft Editor", value=st.session_state.draft_content, height=400)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("💾 Sync Draft"):
            st.session_state.draft_content = edited_draft
            st.success("Draft synchronized.")
    with col2:
        try:
            pdf_bytes = create_pdf(edited_draft)
            st.download_button(label="📄 Export PDF", data=pdf_bytes, file_name="BharatLaw_Draft.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Export Error: {e}")
    st.markdown("---")

# 3. Audio Auto-Play
if st.session_state.audio_to_play:
    st.markdown(autoplay_audio(st.session_state.audio_to_play), unsafe_allow_html=True)
    if st.button("⏹️ Stop Audio"):
        st.session_state.audio_to_play = None
        st.rerun()
    else:
        duration = get_audio_duration(st.session_state.audio_to_play)
        time.sleep(duration + 1.5)
        st.session_state.audio_to_play = None
        st.rerun()

# 4. Handle Pending Query (from Text Input Enter)
if st.session_state.pending_query:
    q = st.session_state.pending_query
    st.session_state.pending_query = None
    process_query(q, use_voice=False)

# 5. Gemini-Style Big Oval Chatbar
st.markdown('<span id="chatbar-marker"></span>', unsafe_allow_html=True)
with st.container():
    # New Column Layout: [Text Input, Model Selector, Mic Button]
    col1, col2, col3 = st.columns([12, 3, 2])
    
    with col1:
        st.text_input(
            "Ask something...", 
            key="chat_input_box", 
            label_visibility="collapsed",
            placeholder="Ask a legal question...",
            on_change=submit_text_query
        )

    with col2:
        AVAILABLE_MODELS = {
            "Llama 4 Scout 17B": "meta-llama/llama-4-scout-17b-16e-instruct",
            "Llama 3.1 8B": "llama-3.1-8b-instant",
            "GPT-OSS 20B": "openai/gpt-oss-20b",
            "Qwen 32B": "qwen/qwen3-32b"
        }
        selected_label = st.selectbox(
            "Model", 
            options=list(AVAILABLE_MODELS.keys()), 
            index=0, 
            label_visibility="collapsed"
        )
        if AVAILABLE_MODELS[selected_label] != st.session_state.current_model:
            st.session_state.current_model = AVAILABLE_MODELS[selected_label]
            st.session_state.agent_system = None
            st.rerun()

    with col3:
        if not st.session_state.audio_to_play:
            voice_text = speech_to_text(
                language='en', 
                start_prompt="Speak", 
                stop_prompt="Stop", 
                just_once=True, 
                use_container_width=True, 
                key='gemini_mic'
            )
            if voice_text and voice_text.strip():
                st.session_state.pending_query = voice_text
                st.rerun()