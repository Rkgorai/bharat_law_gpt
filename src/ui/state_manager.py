import streamlit as st
import os
import shutil
import atexit

def cleanup_recordings():
    dir_path = "db/recordings"
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
        except Exception as e:
            if os.environ.get("BHARAT_LAW_VERBOSE") == "1":
                print(f"[ERROR] Failed to clean recordings dir: {e}")
    try:
        os.makedirs(dir_path, exist_ok=True)
    except Exception:
        pass

def clear_chat_and_recordings():
    st.session_state.messages = []
    st.session_state.audio_to_play = None
    st.session_state.last_played_audio = None
    st.session_state.draft_content = None
    cleanup_recordings()

def initialize_session_state():
    if "theme" not in st.session_state:
        st.session_state.theme = "system"
    if "messages" not in st.session_state:
        cleanup_recordings()
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
    if "pending_query_is_voice" not in st.session_state:
        st.session_state.pending_query_is_voice = False
    if "last_input_was_voice" not in st.session_state:
        st.session_state.last_input_was_voice = False
    if "last_processed_audio_id" not in st.session_state:
        st.session_state.last_processed_audio_id = None
    if "voice_output_all" not in st.session_state:
        st.session_state.voice_output_all = False
    if "last_played_audio" not in st.session_state:
        st.session_state.last_played_audio = None
    if "play_message_content" not in st.session_state:
        st.session_state.play_message_content = None
    if "play_message_index" not in st.session_state:
        st.session_state.play_message_index = None
    if "last_processed_query" not in st.session_state:
        st.session_state.last_processed_query = None

# Register process exit cleanup handler
atexit.register(cleanup_recordings)
