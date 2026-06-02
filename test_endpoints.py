import requests
import json
import sys
import time

BACKEND_URL = "http://localhost:8000"

def log_section(title):
    print("\n" + "="*50)
    print(f"🔍 {title}")
    print("="*50)

def test_health():
    log_section("Testing /api/health Endpoint")
    try:
        start_time = time.time()
        response = requests.get(f"{BACKEND_URL}/api/health", timeout=5)
        latency = (time.time() - start_time) * 1000
        
        print(f"Status Code: {response.status_code}")
        print(f"Latency: {latency:.2f} ms")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy":
                print("✅ Health Check: PASSED")
                return True
        print("❌ Health Check: FAILED")
        return False
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

def test_synthesize():
    log_section("Testing /api/synthesize (TTS) Endpoint")
    payload = {"text": "Hello, this is a test of the Bharat Law Speech Synthesis engine."}
    try:
        start_time = time.time()
        response = requests.post(
            f"{BACKEND_URL}/api/synthesize",
            json=payload,
            timeout=10
        )
        latency = (time.time() - start_time) * 1000
        
        print(f"Status Code: {response.status_code}")
        print(f"Latency: {latency:.2f} ms")
        
        if response.status_code == 200:
            data = response.json()
            audio_url = data.get("audio_url")
            print(f"Generated Audio URL: {audio_url}")
            if audio_url and "/api/static/audio/" in audio_url:
                print("✅ TTS Synthesis: PASSED")
                return True
        print(f"Response: {response.text}")
        print("❌ TTS Synthesis: FAILED")
        return False
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

def test_chat_stream():
    log_section("Testing /api/chat (SSE Stream) Endpoint")
    payload = {
        "query": "What is the penalty for theft under IPC? Keep it very short.",
        "chat_history": [],
        "model": "llama-3.1-8b-instant",
        "bm25_weight": 0.6,
        "vector_weight": 0.4,
        "use_voice": False
    }
    
    headers = {"Accept": "text/event-stream"}
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BACKEND_URL}/api/chat",
            json=payload,
            headers=headers,
            stream=True,
            timeout=15
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code != 200:
            print("❌ Chat Stream Connection: FAILED")
            print(f"Response: {response.text}")
            return False
            
        print("📥 Streaming response from Agent (typewriter output):\n")
        
        event_type = None
        full_text = ""
        
        # Read stream lines
        for line in response.iter_lines():
            if line:
                line_decoded = line.decode('utf-8').strip()
                
                if line_decoded.startswith("event:"):
                    event_type = line_decoded.replace("event:", "").strip()
                elif line_decoded.startswith("data:"):
                    data_str = line_decoded.replace("data:", "").strip()
                    try:
                        data_json = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    
                    if event_type == "token":
                        token = data_json.get("text", "")
                        full_text += token
                        sys.stdout.write(token)
                        sys.stdout.flush()
                    elif event_type == "status":
                        print(f"\n[AGENT STATUS] {data_json.get('msg')}")
                    elif event_type == "draft":
                        print(f"\n[DRAFT GENERATED] Content Length: {len(data_json.get('content'))}")
                    elif event_type == "error":
                        print(f"\n[ERROR] {data_json.get('detail')}")
                    elif event_type == "end":
                        print("\n\n🔚 Stream finished cleanly.")
                        break
        
        total_time = time.time() - start_time
        print(f"\nTotal generation time: {total_time:.2f} seconds")
        
        if full_text:
            print("\n✅ Chat Stream: PASSED")
            return True
        else:
            print("\n❌ Chat Stream: FAILED (Empty stream)")
            return False
            
    except Exception as e:
        print(f"\n❌ Connection Error: {e}")
        return False

def main():
    print("==================================================")
    print("🧪 BHARAT LAW GPT - ENDPOINTS VERIFICATION SYSTEM")
    print("==================================================")
    
    health_ok = test_health()
    tts_ok = test_synthesize()
    chat_ok = test_chat_stream()
    
    log_section("Final Verification Report")
    print(f"1. Health Check Endpoint:  {'🟢 PASSED' if health_ok else '🔴 FAILED'}")
    print(f"2. TTS Synthesis Endpoint: {'🟢 PASSED' if tts_ok else '🔴 FAILED'}")
    print(f"3. Streaming Chat Endpoint:{'🟢 PASSED' if chat_ok else '🔴 FAILED'}")
    print("="*50)
    
    if health_ok and tts_ok and chat_ok:
        print("\n✨ ALL ENDPOINTS WORK CORRECTLY! BACKEND IS READY FOR PRODUCTION!")
        sys.exit(0)
    else:
        print("\n⚠️ SOME VERIFICATIONS FAILED. Check backend container logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
