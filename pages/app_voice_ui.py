# import os
# import sys
# import base64

# # Add the parent directory to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import streamlit as st
# import time
# from streamlit_mic_recorder import speech_to_text
# from src.search import RAGSearch
# from src.voice_handler import VoiceHandler

# # --- CONFIGURATION ---
# PAGE_TITLE = "Bharat Law GPT"
# PAGE_ICON = "🇮🇳"
# DB_PATH = "db/faiss_store"

# st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

# # --- CUSTOM CSS ---
# st.markdown(
#     """
#     <style>
#         .stApp { padding-bottom: 140px; } 
        
#         .user-msg {
#             background-color: #dcf8c6; color: #000; padding: 10px 15px;
#             border-radius: 15px 0 15px 15px; text-align: right;
#             margin: 5px 0 5px auto; max-width: 80%; display: block;
#             box-shadow: 0 1px 1px rgba(0,0,0,0.1); font-family: sans-serif;
#         }
#         .bot-msg {
#             background-color: #f0f2f6; color: #000; padding: 10px 15px;
#             border-radius: 0 15px 15px 15px; text-align: left;
#             margin: 5px auto 5px 0; max-width: 80%; display: block;
#             box-shadow: 0 1px 1px rgba(0,0,0,0.1);
#             font-family: sans-serif;
#         }

#         .footer-container {
#             position: fixed; bottom: 0; left: 0; width: 100%;
#             background-color: #ffffff; padding: 15px;
#             border-top: 1px solid #ddd; z-index: 999;
#             text-align: center;
#         }
        
#         .stButton button { width: 100%; border-radius: 20px; }
        
#         #MainMenu {visibility: hidden;}
#         footer {visibility: hidden;}
#         header {visibility: hidden;}
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # --- INIT STATE ---
# if "history" not in st.session_state:
#     st.session_state.history = []
# if "current_model" not in st.session_state:
#     st.session_state.current_model = "moonshotai/kimi-k2-instruct-0905"
# if "temp_input" not in st.session_state:
#     st.session_state.temp_input = None
# if "audio_to_play" not in st.session_state:
#     st.session_state.audio_to_play = None
# if "last_answer_text" not in st.session_state:
#     st.session_state.last_answer_text = ""

# # --- LOAD ENGINE ---
# @st.cache_resource
# def get_voice_handler():
#     return VoiceHandler()

# @st.cache_resource
# def get_rag_engine(model_name):
#     return RAGSearch(persist_dir=DB_PATH, llm_model=model_name)

# voice = get_voice_handler()
# rag_engine = get_rag_engine(st.session_state.current_model)

# # --- HELPER: AUDIO PLAYER (HIDDEN) ---
# def autoplay_audio(file_path):
#     with open(file_path, "rb") as f:
#         data = f.read()
#     b64 = base64.b64encode(data).decode()
#     # Added style="display:none" to hide the player while keeping functionality
#     md = f"""
#         <audio controls autoplay style="display:none">
#         <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
#         </audio>
#     """
#     return md

# # --- SIDEBAR ---
# with st.sidebar:
#     st.header("⚙️ Settings")
#     if st.button("💬 Switch to Text Mode"):
#         st.switch_page("app_ui.py")
#     st.divider()
    
#     AVAILABLE_MODELS = {
#         "Kimi K2 (Moonshot)": "moonshotai/kimi-k2-instruct-0905",
#         "Llama 3.1 8B": "llama-3.1-8b-instant",
#         "GPT-OSS 20B": "openai/gpt-oss-20b",
#         "Qwen 32B": "qwen/qwen3-32b"
#     }
#     selected_label = st.selectbox("Brain", options=list(AVAILABLE_MODELS.keys()), index=0)
#     selected_model_id = AVAILABLE_MODELS[selected_label]

#     if selected_model_id != st.session_state.current_model:
#         st.session_state.current_model = selected_model_id
#         st.cache_resource.clear()
#         st.toast(f"Switched to {selected_label}", icon="🧠")

#     st.divider()
#     if st.button("🗑️ Clear Chat", use_container_width=True):
#         st.session_state.history = []
#         st.session_state.temp_input = None
#         st.session_state.audio_to_play = None
#         st.rerun()

# # --- MAIN UI ---
# st.markdown("### 🏛️ Bharat Law Assistant")

# if not st.session_state.history and not st.session_state.temp_input:
#     st.info("Tap 'Start Recording' below to begin.")

# # Display Chat
# for msg in st.session_state.history:
#     if msg["role"] == "user":
#         st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
#     else:
#         st.markdown(f"<div class='bot-msg'>{msg['content']}</div>", unsafe_allow_html=True)

# # --- AUDIO PLAYER LOGIC ---
# if st.session_state.audio_to_play:
#     # 1. Show STOP Button
#     col_center = st.columns([1, 1, 1])
#     with col_center[1]:
#         if st.button("⏹️ STOP SPEAKING", use_container_width=True):
#             st.session_state.audio_to_play = None
#             st.rerun() # Interrupts everything and restarts

#     # 2. Render Hidden Player
#     audio_html = autoplay_audio(st.session_state.audio_to_play)
#     st.markdown(audio_html, unsafe_allow_html=True)
    
#     # 3. Estimate Duration & Wait (The Magic Trick)
#     # Approx 2.5 words per second (150 wpm) + 2 seconds buffer
#     word_count = len(st.session_state.last_answer_text.split())
#     estimated_duration = (word_count / 2.5) + 2
    
#     # This sleep keeps the script 'running' so the Stop button stays visible.
#     # If the user clicks Stop, Streamlit interrupts this sleep and reruns immediately.
#     # If the sleep finishes naturally, we clear the audio.
#     time.sleep(estimated_duration)
    
#     # Audio finished naturally -> Clear it
#     st.session_state.audio_to_play = None
#     st.rerun()


# # --- FOOTER ---
# with st.container():
#     st.markdown('<div class="footer-container">', unsafe_allow_html=True)
    
#     if st.session_state.temp_input:
#         edited_text = st.text_area("Review:", value=st.session_state.temp_input, height=100, label_visibility="collapsed")
#         st.write("") 
#         col1, col2 = st.columns(2)
        
#         with col1:
#             if st.button("❌ Cancel", use_container_width=True):
#                 st.session_state.temp_input = None
#                 st.rerun()
                
#         with col2:
#             if st.button("✅ Submit", use_container_width=True):
#                 final_query = edited_text
#                 st.session_state.temp_input = None 
                
#                 st.session_state.history.append({"role": "user", "content": final_query})
                
#                 with st.chat_message("assistant"):
#                      placeholder = st.empty()
#                      placeholder.markdown("🔍 *Searching...*")
                
#                 recent_context = st.session_state.history[-3:]
#                 result = rag_engine.search_and_summarize(query=final_query, chat_history=recent_context)
#                 answer_text = result["answer"]
                
#                 st.session_state.history.append({"role": "assistant", "content": answer_text})
                
#                 # --- GENERATE & QUEUE AUDIO ---
#                 audio_path = voice.synthesize(answer_text)
#                 if audio_path:
#                     st.session_state.audio_to_play = audio_path
#                     st.session_state.last_answer_text = answer_text # Store for duration calc
                
#                 st.rerun()
#     else:
#         # Don't show recorder if audio is playing
#         if st.session_state.audio_to_play:
#             st.info("🔊 Assistant is speaking...")
#         else:
#             new_text = speech_to_text(
#                 language='en', start_prompt="🔴 Start Recording", stop_prompt="⏹️ Stop Recording", 
#                 just_once=True, use_container_width=True, key='RECORDER_WIDGET'
#             )
#             if new_text:
#                 st.session_state.temp_input = new_text
#                 st.rerun()

#     st.markdown('</div>', unsafe_allow_html=True)

import os
import sys
import base64

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import time
# NEW IMPORT
from mutagen.mp3 import MP3
from streamlit_mic_recorder import speech_to_text
from src.search import RAGSearch
from src.voice_handler import VoiceHandler

# --- CONFIGURATION ---
PAGE_TITLE = "Bharat Law GPT"
PAGE_ICON = "🇮🇳"
DB_PATH = "db/faiss_store"

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

# --- CUSTOM CSS ---
st.markdown(
    """
    <style>
        .stApp { padding-bottom: 140px; } 
        
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

        .footer-container {
            position: fixed; bottom: 0; left: 0; width: 100%;
            background-color: #ffffff; padding: 15px;
            border-top: 1px solid #ddd; z-index: 999;
            text-align: center;
        }
        
        .stButton button { width: 100%; border-radius: 20px; }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- INIT STATE ---
if "history" not in st.session_state:
    st.session_state.history = []
if "current_model" not in st.session_state:
    st.session_state.current_model = "moonshotai/kimi-k2-instruct-0905"
if "temp_input" not in st.session_state:
    st.session_state.temp_input = None
if "audio_to_play" not in st.session_state:
    st.session_state.audio_to_play = None

# --- LOAD ENGINE ---
@st.cache_resource
def get_voice_handler():
    return VoiceHandler()

@st.cache_resource
def get_rag_engine(model_name):
    return RAGSearch(persist_dir=DB_PATH, llm_model=model_name)

voice = get_voice_handler()
rag_engine = get_rag_engine(st.session_state.current_model)

# --- HELPER: AUDIO PLAYER (HIDDEN) ---
def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    md = f"""
        <audio controls autoplay style="display:none">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    return md

# --- NEW HELPER: GET EXACT DURATION ---
def get_audio_duration(file_path):
    try:
        audio = MP3(file_path)
        return audio.info.length
    except Exception:
        return 5 # Fallback if file read fails

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("💬 Switch to Text Mode"):
        st.switch_page("app_ui.py")
    st.divider()
    
    AVAILABLE_MODELS = {
        "Kimi K2 (Moonshot)": "moonshotai/kimi-k2-instruct-0905",
        "Llama 3.1 8B": "llama-3.1-8b-instant",
        "GPT-OSS 20B": "openai/gpt-oss-20b",
        "Qwen 32B": "qwen/qwen3-32b"
    }
    selected_label = st.selectbox("Brain", options=list(AVAILABLE_MODELS.keys()), index=0)
    selected_model_id = AVAILABLE_MODELS[selected_label]

    if selected_model_id != st.session_state.current_model:
        st.session_state.current_model = selected_model_id
        st.cache_resource.clear()
        st.toast(f"Switched to {selected_label}", icon="🧠")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.temp_input = None
        st.session_state.audio_to_play = None
        st.rerun()

# --- MAIN UI ---
st.markdown("### 🏛️ Bharat Law Assistant")

if not st.session_state.history and not st.session_state.temp_input:
    st.info("Tap 'Start Recording' below to begin.")

# Display Chat
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>{msg['content']}</div>", unsafe_allow_html=True)

# --- AUDIO PLAYER LOGIC ---
if st.session_state.audio_to_play:
    # 1. Show STOP Button
    col_center = st.columns([1, 1, 1])
    with col_center[1]:
        if st.button("⏹️ STOP SPEAKING", use_container_width=True):
            st.session_state.audio_to_play = None
            st.rerun()

    # 2. Render Hidden Player
    audio_html = autoplay_audio(st.session_state.audio_to_play)
    st.markdown(audio_html, unsafe_allow_html=True)
    
    # 3. Get EXACT Duration & Wait
    exact_duration = get_audio_duration(st.session_state.audio_to_play)
    
    # We add 1.5s buffer for browser loading/network lag
    time.sleep(exact_duration + 1.5)
    
    # Audio finished -> Clear it
    st.session_state.audio_to_play = None
    st.rerun()


# --- FOOTER ---
with st.container():
    st.markdown('<div class="footer-container">', unsafe_allow_html=True)
    
    if st.session_state.temp_input:
        edited_text = st.text_area("Review:", value=st.session_state.temp_input, height=100, label_visibility="collapsed")
        st.write("") 
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.temp_input = None
                st.rerun()
                
        with col2:
            if st.button("✅ Submit", use_container_width=True):
                final_query = edited_text
                st.session_state.temp_input = None 
                
                st.session_state.history.append({"role": "user", "content": final_query})
                
                with st.chat_message("assistant"):
                     placeholder = st.empty()
                     placeholder.markdown("🔍 *Searching...*")
                
                recent_context = st.session_state.history[-3:]
                result = rag_engine.search_and_summarize(query=final_query, chat_history=recent_context)
                answer_text = result["answer"]
                
                st.session_state.history.append({"role": "assistant", "content": answer_text})
                
                # --- GENERATE & QUEUE AUDIO ---
                audio_path = voice.synthesize(answer_text)
                if audio_path:
                    st.session_state.audio_to_play = audio_path
                
                st.rerun()
    else:
        # Don't show recorder if audio is playing
        if st.session_state.audio_to_play:
            st.info("🔊 Assistant is speaking...")
        else:
            new_text = speech_to_text(
                language='en', start_prompt="🔴 Start Recording", stop_prompt="⏹️ Stop Recording", 
                just_once=True, use_container_width=True, key='RECORDER_WIDGET'
            )
            if new_text:
                st.session_state.temp_input = new_text
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)