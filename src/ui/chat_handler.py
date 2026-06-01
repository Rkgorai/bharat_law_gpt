import streamlit as st
import os
import sys
import base64
import time
import shutil
import atexit
from mutagen.mp3 import MP3
from langchain_core.messages import HumanMessage, AIMessage

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.ui.dependencies import get_legal_agent, get_voice_handler
from src.ui.pdf_utils import create_pdf

AVAILABLE_MODELS = {
    "Llama 4 Scout 17B": "meta-llama/llama-4-scout-17b-16e-instruct",
    "Llama 3.1 8B": "llama-3.1-8b-instant",
    "GPT-OSS 20B": "openai/gpt-oss-20b",
    "Qwen 32B": "qwen/qwen3-32b"
}

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
        # Check for our invisible zero-width space voice metadata token
        is_voice = query.endswith('\u200b')
        clean_query = query.replace('\u200b', '').strip()
        
        # Prevent double execution of the exact same query in rapid succession
        if st.session_state.get("last_processed_query") == clean_query:
            st.session_state.chat_input_box = "" # Clear duplicate input
            return
            
        # Set the lock immediately in the submission event to block concurrent events!
        st.session_state.last_processed_query = clean_query
            
        st.session_state.pending_query = clean_query
        st.session_state.pending_query_is_voice = is_voice
        st.session_state.chat_input_box = "" # Clear input
        st.session_state.last_input_was_voice = False

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

def process_query(query: str, use_voice: bool = False):
    # Reset any previous audio states
    st.session_state.audio_to_play = None
    st.session_state.last_played_audio = None
    st.session_state.play_message_content = None
    st.session_state.last_processed_query = query

    print(f"[USER] Query: \"{query}\"")
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
        
        if use_voice:
            query_for_llm = query + "\n\n(Important instruction: I am listening via voice output. Keep your answer simple, direct, and concise so it is easy to listen to. Do not include detailed explanations or long paragraphs.)"
        else:
            query_for_llm = query
            
        formatted_history.append(HumanMessage(content=query_for_llm))
        
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
                                
            # First, collapse the thinking status container cleanly and instantly
            status_container.update(label="Complete", state="complete", expanded=False)
            
            if answer_text:
                # 1. Render the text response instantly
                message_placeholder = st.empty()
                message_placeholder.markdown(answer_text)
                
                # 2. Append to messages history immediately (audio_path=None initially)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer_text,
                    "audio_path": None
                })
                
                # 3. Perform text-to-speech synthesis (synchronous, takes 2-3 seconds)
                audio_path = None
                try:
                    voice_handler = get_voice_handler()
                    audio_path = voice_handler.synthesize(answer_text)
                    # Cache the synthesized audio path in the saved history message
                    st.session_state.messages[-1]["audio_path"] = audio_path
                except Exception as tts_err:
                    print(f"[ERROR] Pre-synthesis failed: {tts_err}")

                # 4. If voice mode is active, trigger autoplay immediately!
                if use_voice and audio_path:
                    st.session_state.audio_to_play = audio_path
                    st.rerun()
                        
        except Exception as e:
            status_container.update(label="Error Occurred", state="error", expanded=True)
            st.error(f"Error: {e}")
        finally:
            # Release the lock so the user can query again later if needed
            st.session_state.last_processed_query = None

def render_chat_history():
    if not st.session_state.messages:
        st.info("👋 Welcome! Type or speak a legal question to begin.")
        
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # Render text in full width so it aligns 100% perfectly with user chat text!
                st.markdown(message["content"])
                
                audio_path = message.get("audio_path")
                
                # Check if this specific audio is currently playing
                is_playing = (
                    audio_path is not None and 
                    st.session_state.get("audio_to_play") == audio_path
                )
                
                if is_playing:
                    # Show Stop icon (black & white normal Unicode symbol)
                    if st.button("■", key=f"stop_voice_{idx}", help="Stop listening"):
                        st.session_state.audio_to_play = None
                        st.session_state.last_played_audio = None
                        st.rerun()
                else:
                    # Show Play icon (black & white normal Unicode symbol)
                    if st.button("▶", key=f"play_voice_{idx}", help="Listen to this message"):
                        if audio_path:
                            st.session_state.audio_to_play = audio_path
                            st.session_state.last_played_audio = None
                            st.rerun()
                        else:
                            # Fallback generation on the fly if pre-synthesis was not cached yet
                            st.session_state.play_message_content = message["content"]
                            st.session_state.play_message_index = idx
                            st.rerun()
            else:
                st.markdown(message["content"])

def render_draft_editor():
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

def handle_autoplay_audio():
    if st.session_state.audio_to_play:
        # Check if we have already played this specific audio file in this session run to avoid looping autoplay
        if st.session_state.get("last_played_audio") != st.session_state.audio_to_play:
            st.markdown(autoplay_audio(st.session_state.audio_to_play), unsafe_allow_html=True)
            st.session_state.last_played_audio = st.session_state.audio_to_play

def handle_message_playback():
    if st.session_state.get("play_message_content"):
        content = st.session_state.play_message_content
        idx = st.session_state.get("play_message_index")
        st.session_state.play_message_content = None
        st.session_state.play_message_index = None
        
        # Stop any currently playing audio
        st.session_state.audio_to_play = None
        st.session_state.last_played_audio = None
        
        with st.spinner("Preparing voice audio..."):
            voice_handler = get_voice_handler()
            audio_path = voice_handler.synthesize(content)
            if audio_path:
                # Cache it in the message dictionary so we don't have to generate it again!
                if idx is not None and idx < len(st.session_state.messages):
                    st.session_state.messages[idx]["audio_path"] = audio_path
                st.session_state.audio_to_play = audio_path
                st.rerun()

def handle_pending_query():
    if st.session_state.pending_query:
        q = st.session_state.pending_query
        is_voice = st.session_state.get('pending_query_is_voice', False)
        st.session_state.pending_query = None
        st.session_state.pending_query_is_voice = False
        process_query(q, use_voice=is_voice)

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

# Register process exit cleanup handler
atexit.register(cleanup_recordings)
