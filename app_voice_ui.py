import streamlit as st
import time
import threading
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
        
        /* Chat Bubbles */
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

        /* Footer Container */
        .footer-container {
            position: fixed; bottom: 0; left: 0; width: 100%;
            background-color: #ffffff; padding: 15px;
            border-top: 1px solid #ddd; z-index: 999;
            text-align: center;
        }
        
        /* Stop Button Styling */
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
if "is_speaking" not in st.session_state:
    st.session_state.is_speaking = False
if "temp_input" not in st.session_state:
    st.session_state.temp_input = None

# --- LOAD ENGINE ---
@st.cache_resource
def get_voice_handler():
    return VoiceHandler()

@st.cache_resource
def get_rag_engine(model_name):
    return RAGSearch(persist_dir=DB_PATH, llm_model=model_name)

voice = get_voice_handler()
rag_engine = get_rag_engine(st.session_state.current_model)

# --- AUTO-RESET LOGIC ---
# If the app reloads and audio is NOT playing, force the flag to False immediately.
if st.session_state.is_speaking and not voice.is_playing():
    st.session_state.is_speaking = False

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    
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
        voice.stop_audio()
        st.session_state.is_speaking = False
        st.rerun()

# --- MAIN UI: HISTORY ---
st.markdown("### 🏛️ Bharat Law Assistant")

if not st.session_state.history and not st.session_state.temp_input:
    st.info("Tap 'Start Recording' below to begin.")

for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>{msg['content']}</div>", unsafe_allow_html=True)


# --- AUDIO CONTROL (Floating Stop Button) ---
# We use the flag + the actual status from the handler
if st.session_state.is_speaking and voice.is_playing():
    col_center = st.columns([1, 1, 1])
    with col_center[1]:
        if st.button("⏹️ STOP SPEAKING", key="stop_speaking_btn", use_container_width=True):
            voice.stop_audio()
            st.session_state.is_speaking = False
            st.rerun()

# --- FOOTER: LOGIC MACHINE ---
with st.container():
    st.markdown('<div class="footer-container">', unsafe_allow_html=True)
    
    # STATE A: Captured Text -> Show Edit/Submit/Cancel
    if st.session_state.temp_input:
        edited_text = st.text_area(
            "Review & Edit:", 
            value=st.session_state.temp_input, 
            height=100,
            label_visibility="collapsed"
        )
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
                
                # Update History
                st.session_state.history.append({"role": "user", "content": final_query})
                
                # Show Searching
                with st.chat_message("assistant"):
                     placeholder = st.empty()
                     placeholder.markdown("🔍 *Searching...*")
                
                # Search
                recent_context = st.session_state.history[-3:]
                result = rag_engine.search_and_summarize(query=final_query, chat_history=recent_context)
                answer_text = result["answer"]
                
                # Save & Speak
                st.session_state.history.append({"role": "assistant", "content": answer_text})
                
                def play_audio_thread():
                    voice.stream_audio(answer_text)
                
                st.session_state.is_speaking = True
                audio_thread = threading.Thread(target=play_audio_thread)
                audio_thread.start()
                
                # Rerun immediately to show the "Stop" button
                time.sleep(0.1)
                st.rerun()

    # STATE B: Recorder Button
    else:
        # Disable recorder if we are currently speaking to prevent overlap
        if st.session_state.is_speaking:
            st.info("🔊 Assistant is speaking...")
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

# --- WATCHER LOOP---
# This block runs at the end of the script.
# If we are speaking, it waits 1s, then reruns the whole script.
# This forces the "if voice.is_playing()" check at the top to run again.
if st.session_state.is_speaking:
    time.sleep(1)
    st.rerun()