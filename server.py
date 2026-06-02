import os
import json
import asyncio
import uuid
import shutil
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import core modules
from src.agent.agent import create_legal_agent
import src.agent.agent as agent_module
from src.agent.guardrails import QueryGuardrail
from src.agent.pii_protector import PIIProtector
from src.voice.voice_handler import VoiceHandler

# Initialize FastAPI App
app = FastAPI(
    title="Bharat Law GPT - API Server",
    description="Production-grade decoupled backend for Indian Legal AI Assistant.",
    version="1.0.0"
)

# Configure CORS for backend-frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the Streamlit host
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Ensure database and cache directories exist
os.makedirs("db/tts_cache", exist_ok=True)
os.makedirs("db/recordings", exist_ok=True)

# Mount TTS Audio Cache as Static Files
app.mount("/api/static/audio", StaticFiles(directory="db/tts_cache"), name="audio")

# Initialize global managers
guardrail = QueryGuardrail()
pii_protector = PIIProtector()
voice_handler = VoiceHandler()

# Process-level agent cache to avoid re-instantiating agents
_AGENT_CACHE = {}

def get_cached_agent(model_name: str):
    global _AGENT_CACHE
    if model_name not in _AGENT_CACHE:
        if os.environ.get("BHARAT_LAW_VERBOSE") == "1":
            print(f"[API] Loading and caching agent for model: {model_name}")
        _AGENT_CACHE[model_name] = create_legal_agent(llm_model=model_name)
    return _AGENT_CACHE[model_name]

# Pydantic models for chat requests
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    chat_history: List[ChatMessage] = []
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    bm25_weight: float = 0.6
    vector_weight: float = 0.4
    use_voice: bool = False

@app.get("/api/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "database": "loaded"}

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Endpoint to transcribe recorded audio bytes from the Streamlit frontend.
    """
    try:
        temp_file_path = f"db/recordings/upload_{uuid.uuid4().hex}.wav"
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        # Read the file bytes and transcribe
        with open(temp_file_path, "rb") as f:
            audio_bytes = f.read()
            
        transcribed_text = voice_handler.transcribe(audio_bytes)
        
        # Cleanup temp upload file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
        if not transcribed_text:
            raise HTTPException(status_code=400, detail="Transcription failed.")
            
        return {"text": transcribed_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/synthesize")
async def synthesize_text(payload: Dict[str, str]):
    """
    Endpoint to synthesize legal text to Edge-TTS audio.
    """
    text = payload.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text payload is empty.")
    try:
        audio_path = voice_handler.synthesize(text)
        if not audio_path:
            raise HTTPException(status_code=500, detail="Audio synthesis failed.")
            
        filename = os.path.basename(audio_path)
        # Expose static URL of the file
        audio_url = f"/api/static/audio/{filename}"
        return {"audio_url": audio_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def agent_stream_generator(request: ChatRequest):
    """
    Generates an SSE stream of LangGraph events, injecting PII masking,
    Guardrails, dynamic weights, and de-anonymization.
    """
    # 1. Run Query Guardrails
    is_safe, refusal_msg = guardrail.check_query(request.query)
    if not is_safe:
        yield f"event: token\ndata: {json.dumps({'text': refusal_msg})}\n\n"
        yield "event: end\ndata: {}\n\n"
        return

    # 2. Apply PII Anonymization
    anonymized_query, pii_mapping = pii_protector.anonymize(request.query)
    
    # 3. Dynamic RAG weights assignment
    if agent_module._local_search is None:
        agent_module._local_search = agent_module.LocalSearchWrapper()
    agent_module._local_search.vectorstore.bm25_weight = request.bm25_weight
    agent_module._local_search.vectorstore.vector_weight = request.vector_weight
    
    # 4. Map chat history to LangChain messages format
    from langchain_core.messages import HumanMessage, AIMessage
    formatted_history = []
    for msg in request.chat_history:
        # Anonymize history to keep privacy uniform
        anon_hist_content, _ = pii_protector.anonymize(msg.content)
        if msg.role == "user":
            formatted_history.append(HumanMessage(content=anon_hist_content))
        else:
            formatted_history.append(AIMessage(content=anon_hist_content))
            
    # Add voice instruction injection if required
    final_query = anonymized_query
    if request.use_voice:
        final_query += "\n\n(Important instruction: I am listening via voice output. Keep your answer simple, direct, and concise so it is easy to listen to. Do not include detailed explanations or long paragraphs.)"
        
    formatted_history.append(HumanMessage(content=final_query))
    
    # 5. Fetch Agent
    try:
        agent = get_cached_agent(request.model)
    except Exception as err:
        yield f"event: error\ndata: {json.dumps({'detail': f'Failed to initialize model: {str(err)}'})}\n\n"
        return

    # 6. Stream LangGraph Agent loop
    try:
        # We process event streams natively
        loop = asyncio.get_event_loop()
        
        # Helper to execute stream synchronously inside a thread pool to avoid blocking the async event loop
        def get_stream():
            return agent.stream({"messages": formatted_history})
            
        stream = await loop.run_in_executor(None, get_stream)
        
        for event in stream:
            for node_name, value in event.items():
                if "messages" in value:
                    messages = value["messages"]
                    if not isinstance(messages, list):
                        messages = [messages]
                        
                    for msg in messages:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                tool_name = tool_call.get("name", "tool")
                                yield f"event: status\ndata: {json.dumps({'msg': f'⚙️ Running `{tool_name}`'})}\n\n"
                                
                        # Yield tool extraction result
                        elif getattr(msg, "type", "") == "tool":
                            yield f"event: status\ndata: {json.dumps({'msg': f'✅ Extracted data from `{msg.name}`'})}\n\n"
                            
                            # Catch generated drafts & stream it to UI
                            if "<DRAFT_SAVED>" in str(msg.content):
                                if os.path.exists("db/latest_draft.txt"):
                                    with open("db/latest_draft.txt", "r") as f:
                                        draft_text = f.read()
                                    # De-anonymize the draft text!
                                    clean_draft = pii_protector.deanonymize(draft_text, pii_mapping)
                                    yield f"event: draft\ndata: {json.dumps({'content': clean_draft})}\n\n"
                                    
                        # Yield live AI answer token chunks
                        if getattr(msg, "type", "") == "ai" and msg.content:
                            # De-anonymize token chunk on-the-fly!
                            clean_chunk = pii_protector.deanonymize(msg.content, pii_mapping)
                            yield f"event: token\ndata: {json.dumps({'text': clean_chunk})}\n\n"
                            
        yield "event: end\ndata: {}\n\n"
    except Exception as e:
        err_msg = str(e)
        if "tool_use_failed" in err_msg or "Failed to call a function" in err_msg:
            fallback_msg = (
                "⚠️ **Assistant Note**: I encountered a minor formatting alignment issue while attempting to draft your document. "
                "To ensure I generate a complete agreement without any placeholders or missing facts, could you please provide or verify the following details: \n\n"
                "- **Borrower Name & Address**\n"
                "- **Lender Name & Address**\n"
                "- **Loan Amount & Interest Rate**\n"
                "- **Loan Term & Repayment Dates**\n\n"
                "Please reply with these details, and I will compile them into a professional draft right away!"
            )
            yield f"event: token\ndata: {json.dumps({'text': fallback_msg})}\n\n"
        elif "rate_limit" in err_msg.lower():
            fallback_msg = "⏳ **System Note**: The model is currently experiencing high demand. Please wait a moment and try submitting your request again."
            yield f"event: token\ndata: {json.dumps({'text': fallback_msg})}\n\n"
        else:
            fallback_msg = (
                f"🔍 **Assistant Note**: I encountered an unexpected error while processing your request: *{err_msg}*.\n\n"
                "Please try adjusting your query slightly or re-submitting your details so I can assist you."
            )
            yield f"event: token\ndata: {json.dumps({'text': fallback_msg})}\n\n"
        yield "event: end\ndata: {}\n\n"

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    SSE Streaming endpoint for legal queries.
    """
    return StreamingResponse(
        agent_stream_generator(request),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
