import streamlit as st
import os
import base64
from streamlit_mic_recorder import speech_to_text
from src.search import RAGSearch
from src.voice_handler import VoiceHandler

# --- CONFIGURATION ---
PAGE_TITLE = "Bharat Law GPT (Live Voice)"
PAGE_ICON = "🎙️"
DB_PATH = "db/faiss_store"

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# --- CUSTOM CSS ---
st.markdown(
    """
    <style>
        .stApp { background-color: #f8f9fa; }
        .main-header { text-align: center; color: #2c3e50; margin-bottom: 2rem; }
        .chat-container { max-width: 800px; margin: 0 auto; }
        .user-bubble { 
            background-color: #e3f2fd; color: #0d47a1; 
            padding: 15px; border-radius: 20px 20px 0 20px; 
            margin: 10px 0; text-align: right; box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .bot-bubble { 
            background-color: #ffffff; color: #333; 
            padding: 15px; border-radius: 20px 20px 20px 0; 
            border-left: 5px solid #ff9800;
            margin: 10px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- SESSION STATE ---
if "history" not in st.session_state:
    st.session_state.history = []
if "current_model" not in st.session_state:
    st.session_state.current_model = "llama-3.1-8b-instant"

# --- HELPER: HTML AUDIO (Fixes ID/Key Errors) ---
def render_audio_html(file_path):
    """
    Renders an audio player using HTML5, bypassing Streamlit's widget ID system.
    This fixes the 'Duplicate ID' and 'Unexpected Argument' errors.
    """
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls preload="none">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        return md
    except Exception as e:
        return f"Error loading audio: {e}"

# --- LOAD RESOURCES ---
@st.cache_resource
def get_voice_handler():
    return VoiceHandler()

@st.cache_resource
def get_rag_engine(persist_dir, model_name):
    return RAGSearch(persist_dir=persist_dir, llm_model=model_name)

voice = get_voice_handler()

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Brain Settings")
    
    AVAILABLE_MODELS = {
        "Llama 3.1 8B (Fastest)": "llama-3.1-8b-instant",
        "Kimi K2 (Moonshot)": "moonshotai/kimi-k2-instruct-0905",
        "GPT-OSS 20B (Accurate)": "openai/gpt-oss-20b",
        "Qwen 32B (Powerful)": "qwen/qwen3-32b"
    }
    
    selected_label = st.selectbox("Select Model", options=list(AVAILABLE_MODELS.keys()), index=0)
    selected_model_id = AVAILABLE_MODELS[selected_label]

    if selected_model_id != st.session_state.current_model:
        st.session_state.current_model = selected_model_id
        st.cache_resource.clear()
        st.toast(f"Switched to {selected_label}", icon="🧠")

    st.divider()
    st.info("🎙️ **Instructions:**\n\n1. Tap the Mic.\n2. Speak.\n3. Stop speaking to send.")
    
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# --- MAIN CONTENT ---
st.markdown("<div class='main-header'><h1>🎙️ Bharat Law Live</h1><p>Hands-free Legal Assistant</p></div>", unsafe_allow_html=True)

# --- 1. VOICE INPUT ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    text_input = speech_to_text(
        language='en',
        start_prompt="🔴 TAP TO SPEAK",
        stop_prompt="⏹️ LISTENING... (Stop speaking to send)",
        just_once=True,
        use_container_width=True,
        key='STT'
    )

# --- 2. LOGIC ---
if text_input:
    last_user_msg = ""
    if len(st.session_state.history) > 0:
        for msg in reversed(st.session_state.history):
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break
    
    if text_input != last_user_msg:
        st.session_state.history.append({"role": "user", "content": text_input})
        
        with st.status("🧠 Consulting Constitution...", expanded=True) as status:
            rag_engine = get_rag_engine(DB_PATH, st.session_state.current_model)
            recent_context = st.session_state.history[-3:]
            result = rag_engine.search_and_summarize(query=text_input, chat_history=recent_context)
            answer_text = result["answer"]
            
            status.update(label="🗣️ Synthesizing Voice...", state="running")
            audio_path = voice.synthesize(answer_text)
            
            status.update(label="✅ Ready!", state="complete", expanded=False)

        st.session_state.history.append({
            "role": "assistant", 
            "content": answer_text,
            "audio": audio_path
        })
        st.rerun()

# --- 3. DISPLAY CHAT ---
with st.container():
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-bubble'>👤 {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-bubble'>⚖️ {msg['content']}</div>", unsafe_allow_html=True)
            
            if "audio" in msg and msg["audio"]:
                # FIXED: Using HTML Audio to avoid ID conflicts and Key errors
                audio_html = render_audio_html(msg["audio"])
                st.markdown(audio_html, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)