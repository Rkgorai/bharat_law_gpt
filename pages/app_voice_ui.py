import os
import sys
import base64
import time
import threading

# --- PATH SETUP ---
# Add the parent directory to sys.path so we can import 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from mutagen.mp3 import MP3
from streamlit_mic_recorder import speech_to_text

# IMPORT SHARED LOADERS (Crucial for performance)
from src.shared import get_voice_handler, get_rag_engine

# --- CONFIGURATION ---
PAGE_TITLE = "Bharat Law - Voice Chat"
PAGE_ICON = "🎙️"
DB_PATH = "db/faiss_store"

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

# --- CUSTOM CSS ---
st.markdown(
    """
    <style>
        /* 1. Spacing for Fixed Footer */
        .stApp { padding-bottom: 140px; } 
        
        /* 2. Chat Bubbles */
        .user-msg {
            background-color: #dcf8c6; color: #000; padding: 10px 15px;
            border-radius: 15px 0 15px 15px; text-align: right;
            margin: 5px 0 5px auto; max-width: 80%; display: block;
            box-shadow: 0 1px 1px rgba(0,0,0,0.1); font-family: sans-serif;
        }
        .bot-msg {
            background-color: #f0f2f6; color: #000; padding: 10px 15px;
            border-radius: 0 15px 15px 15px; text-align: left;
            margin: 5px auto 5px 0; max-width: 80%; display: block;
            box-shadow: 0 1px 1px rgba(0,0,0,0.1);
            font-family: sans-serif;
        }

        /* 3. Footer Container */
        .footer-container {
            position: fixed; bottom: 0; left: 0; width: 100%;
            background-color: #ffffff; padding: 15px;
            border-top: 1px solid #ddd; z-index: 999;
            text-align: center;
        }
        
        /* 4. Button Styling */
        .stButton button { width: 100%; border-radius: 20px; }
        
        /* 5. Clean UI */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- INIT SESSION STATE ---
if "history" not in st.session_state:
    st.session_state.history = []
if "current_model" not in st.session_state:
    st.session_state.current_model = "meta-llama/llama-4-scout-17b-16e-instruct"
if "temp_input" not in st.session_state:
    st.session_state.temp_input = None
if "audio_to_play" not in st.session_state:
    st.session_state.audio_to_play = None

# --- LOAD RESOURCES (SHARED) ---
voice = get_voice_handler()
rag_engine = get_rag_engine(DB_PATH, st.session_state.current_model)

# --- HELPER FUNCTIONS ---
def autoplay_audio(file_path):
    """Generates HTML for a hidden audio player that starts automatically."""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    md = f"""
        <audio controls autoplay style="display:none">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    return md

def get_audio_duration(file_path):
    """Gets the exact duration of the MP3 file for auto-hiding logic."""
    try:
        audio = MP3(file_path)
        return audio.info.length
    except Exception:
        return 5 # Fallback safety

# --- SIDEBAR (Navigation & Settings) ---
with st.sidebar:
    # 1. Navigation Buttons
    if st.button("🏠 Back to Home"):
        st.switch_page("app_ui.py")
        
    st.header("⚙️ Settings")
    
    if st.button("💬 Switch to Text Mode"):
        st.switch_page("pages/app_text_ui.py")
    
    st.divider()
    
    # 2. Model Selection
    AVAILABLE_MODELS = {
        "Llama 4 Scout 17B (Instruct)": "meta-llama/llama-4-scout-17b-16e-instruct",
        "Llama 3.1 8B": "llama-3.1-8b-instant",
        "GPT-OSS 20B": "openai/gpt-oss-20b",
        "Qwen 32B": "qwen/qwen3-32b"
    }
    selected_label = st.selectbox("Brain", options=list(AVAILABLE_MODELS.keys()), index=0)
    selected_model_id = AVAILABLE_MODELS[selected_label]

    if selected_model_id != st.session_state.current_model:
        st.session_state.current_model = selected_model_id
        # Clearing cache forces the shared loader to get the new model next time
        st.cache_resource.clear() 
        st.toast(f"Switched to {selected_label}", icon="🧠")

    st.divider()
    
    # 3. Clear Chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.temp_input = None
        st.session_state.audio_to_play = None
        st.rerun()

# --- MAIN UI: CHAT HISTORY ---
st.markdown("### 🎙️ Bharat Law Voice Assistant")

if not st.session_state.history and not st.session_state.temp_input:
    st.info("Tap 'Start Recording' below to begin.")

for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>{msg['content']}</div>", unsafe_allow_html=True)

# --- AUDIO PLAYER LOGIC ---
if st.session_state.audio_to_play:
    # 1. Show STOP Button (Floating Center)
    col_center = st.columns([1, 1, 1])
    with col_center[1]:
        if st.button("⏹️ STOP SPEAKING", use_container_width=True):
            st.session_state.audio_to_play = None
            st.rerun()

    # 2. Render Hidden Player (This starts the sound)
    audio_html = autoplay_audio(st.session_state.audio_to_play)
    st.markdown(audio_html, unsafe_allow_html=True)
    
    # 3. Auto-Hide Logic
    # Get EXACT Duration of the file
    exact_duration = get_audio_duration(st.session_state.audio_to_play)
    
    # Wait for audio to finish (+1.5s buffer for browser loading)
    time.sleep(exact_duration + 1.5)
    
    # Audio finished -> Clear state & Rerun to hide button
    st.session_state.audio_to_play = None
    st.rerun()


# --- FOOTER: RECORDING & EDITING ---
with st.container():
    st.markdown('<div class="footer-container">', unsafe_allow_html=True)
    
    # STATE A: Text Captured -> Show Review/Edit/Submit
    if st.session_state.temp_input:
        # Editable Text Area
        edited_text = st.text_area(
            "Review:", 
            value=st.session_state.temp_input, 
            height=100, 
            label_visibility="collapsed"
        )
        st.write("") # Spacer
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.temp_input = None
                st.rerun()
                
        with col2:
            if st.button("✅ Submit", use_container_width=True):
                final_query = edited_text
                st.session_state.temp_input = None 
                
                # 1. Update History
                st.session_state.history.append({"role": "user", "content": final_query})
                
                # 2. Show Processing UI
                with st.chat_message("assistant"):
                     placeholder = st.empty()
                     placeholder.markdown("🔍 *Searching Legal Database...*")
                
                # 3. RAG Search
                recent_context = st.session_state.history[-3:]
                result = rag_engine.search_and_summarize(query=final_query, chat_history=recent_context)
                answer_text = result["answer"]
                
                # 4. Save Answer
                st.session_state.history.append({"role": "assistant", "content": answer_text})
                
                # 5. Generate Audio & Queue for Playback
                audio_path = voice.synthesize(answer_text)
                if audio_path:
                    st.session_state.audio_to_play = audio_path
                
                st.rerun()

    # STATE B: Ready to Record
    else:
        # Don't show recorder if audio is currently playing (prevents conflict)
        if st.session_state.audio_to_play:
            st.info("🔊 Speaking...")
        else:
            new_text = speech_to_text(
                language='en', 
                start_prompt="🔴 Start Recording", 
                stop_prompt="⏹️ Stop Recording", 
                just_once=True, 
                use_container_width=True, 
                key='RECORDER_WIDGET'
            )
            
            if new_text:
                st.session_state.temp_input = new_text
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)