import streamlit as st
import os
import sys
import base64
import time
import requests
import json

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.ui.pdf_utils import create_pdf

# Fetch Backend URL from environment
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

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
        st.session_state.messages = []
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
    if "bm25_weight" not in st.session_state:
        st.session_state.bm25_weight = 0.6
    if "vector_weight" not in st.session_state:
        st.session_state.vector_weight = 0.4

def autoplay_audio(audio_url):
    """Render HTML5 audio tag pointing directly to backend audio endpoint."""
    return f'<audio controls autoplay style="display:none"><source src="{audio_url}" type="audio/mp3"></audio>'

def ensure_system_ready():
    """FastAPI agent endpoint does not require local startup."""
    pass

def submit_text_query():
    query = st.session_state.chat_input_box
    if query.strip():
        # Check if this is an explicit submit (ends with voice token \u200b or submit token \u200c)
        is_explicit = query.endswith('\u200b') or query.endswith('\u200c')
        
        if not is_explicit:
            # Focus lost (blur event) - DO NOT SUBMIT!
            # Keep the text in the input box so the user can continue typing when they return.
            return
            
        # Clean the query of all invisible tokens
        clean_query = query.replace('\u200b', '').replace('\u200c', '').strip()
        is_voice = query.endswith('\u200b')
        
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
    
    with st.chat_message("user"):
        st.markdown(query)
        
    with st.chat_message("assistant"):
        status_container = st.status("Initializing Backend Connection...", expanded=True)
        answer_text = ""
        
        # Prepare Request Payload
        history_for_api = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]
        
        payload = {
            "query": query,
            "chat_history": history_for_api,
            "model": st.session_state.current_model,
            "bm25_weight": st.session_state.bm25_weight,
            "vector_weight": st.session_state.vector_weight,
            "use_voice": use_voice
        }
        
        try:
            # Connect to FastAPI SSE Stream
            headers = {"Accept": "text/event-stream"}
            response = requests.post(
                f"{BACKEND_URL}/api/chat",
                json=payload,
                headers=headers,
                stream=True,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Backend returned status code {response.status_code}")
                
            status_container.update(label="🤖 Agent Running", state="running")
            message_placeholder = st.empty()
            
            event_type = None
            
            # Read SSE stream chunk-by-chunk
            for line in response.iter_lines():
                if line:
                    line_decoded = line.decode('utf-8').strip()
                    
                    if line_decoded.startswith("event:"):
                        event_type = line_decoded.replace("event:", "").strip()
                    elif line_decoded.startswith("data:"):
                        data_str = line_decoded.replace("data:", "").strip()
                        data_json = json.loads(data_str)
                        
                        if event_type == "token":
                            # Stream plane text tokens with live typewriter effect
                            answer_text += data_json.get("text", "")
                            message_placeholder.markdown(answer_text)
                            
                        elif event_type == "status":
                            # Update tool running statuses live in container
                            status_container.write(data_json.get("msg", ""))
                            
                        elif event_type == "draft":
                            # Capture draft directly from API response stream
                            st.session_state.draft_content = data_json.get("content", "")
                            
                        elif event_type == "error":
                            status_container.error(data_json.get("detail", "Error generated in agent run."))
                            
                        elif event_type == "end":
                            break
                            
            # Update status container to complete
            status_container.update(label="Complete", state="complete", expanded=False)
            
            if answer_text:
                # Append finalized message to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer_text,
                    "audio_path": None
                })
                
                # Pre-synthesize TTS response on the backend
                audio_path = None
                try:
                    resp = requests.post(f"{BACKEND_URL}/api/synthesize", json={"text": answer_text})
                    if resp.status_code == 200:
                        audio_url = f"{BACKEND_URL}{resp.json().get('audio_url')}"
                        st.session_state.messages[-1]["audio_path"] = audio_url
                        audio_path = audio_url
                except Exception as tts_err:
                    print(f"[ERROR] Pre-synthesis failed: {tts_err}")

                # Autoplay if voice mode triggered
                if use_voice and audio_path:
                    st.session_state.audio_to_play = audio_path
                    st.rerun()
                    
            elif st.session_state.draft_content:
                # If we got a draft, reload page to load editor
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "I have generated the legal draft. Please review and export it in the editor below.",
                    "audio_path": None
                })
                st.rerun()
                        
        except Exception as e:
            status_container.update(label="Error Occurred", state="error", expanded=True)
            st.error(f"Failed to communicate with API server: {e}")
        finally:
            st.session_state.last_processed_query = None

def render_chat_history():
    if not st.session_state.messages:
        st.info("👋 Welcome! Type or speak a legal question to begin.")
        
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"])
                
                audio_path = message.get("audio_path")
                
                # Check if this specific audio is playing
                is_playing = (
                    audio_path is not None and 
                    st.session_state.get("audio_to_play") == audio_path
                )
                
                if is_playing:
                    if st.button("■", key=f"stop_voice_{idx}", help="Stop listening"):
                        st.session_state.audio_to_play = None
                        st.session_state.last_played_audio = None
                        st.rerun()
                else:
                    if st.button("▶", key=f"play_voice_{idx}", help="Listen to this message"):
                        if audio_path:
                            st.session_state.audio_to_play = audio_path
                            st.session_state.last_played_audio = None
                            st.rerun()
                        else:
                            # Trigger generation on backend
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
        if st.session_state.get("last_played_audio") != st.session_state.audio_to_play:
            st.markdown(autoplay_audio(st.session_state.audio_to_play), unsafe_allow_html=True)
            st.session_state.last_played_audio = st.session_state.audio_to_play

def handle_message_playback():
    if st.session_state.get("play_message_content"):
        content = st.session_state.play_message_content
        idx = st.session_state.get("play_message_index")
        st.session_state.play_message_content = None
        st.session_state.play_message_index = None
        
        st.session_state.audio_to_play = None
        st.session_state.last_played_audio = None
        
        with st.spinner("Preparing voice audio on backend..."):
            try:
                resp = requests.post(f"{BACKEND_URL}/api/synthesize", json={"text": content})
                if resp.status_code == 200:
                    audio_url = f"{BACKEND_URL}{resp.json().get('audio_url')}"
                    if idx is not None and idx < len(st.session_state.messages):
                        st.session_state.messages[idx]["audio_path"] = audio_url
                    st.session_state.audio_to_play = audio_url
                    st.rerun()
            except Exception as e:
                st.error(f"Voice generation failed: {e}")

def handle_pending_query():
    if st.session_state.pending_query:
        q = st.session_state.pending_query
        is_voice = st.session_state.get('pending_query_is_voice', False)
        st.session_state.pending_query = None
        st.session_state.pending_query_is_voice = False
        process_query(q, use_voice=is_voice)

def clear_chat_and_recordings():
    st.session_state.messages = []
    st.session_state.audio_to_play = None
    st.session_state.last_played_audio = None
    st.session_state.draft_content = None
