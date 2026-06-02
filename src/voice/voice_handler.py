import os
import asyncio
import edge_tts
import uuid
import hashlib

class VoiceHandler:
    def __init__(self):
        if os.environ.get("BHARAT_LAW_VERBOSE") == "1":
            print("[INFO] Initializing Voice Models...")
        self.stt_model = None
        self.tts_voice = "en-IN-NeerjaNeural" 

    def transcribe(self, audio_bytes):
        os.makedirs("db/recordings", exist_ok=True)
        temp_filename = f"db/recordings/temp_{uuid.uuid4().hex}.wav"
        with open(temp_filename, "wb") as f:
            f.write(audio_bytes)
        try:
            if self.stt_model is None:
                from faster_whisper import WhisperModel
                if os.environ.get("BHARAT_LAW_VERBOSE") == "1":
                    print("[INFO] Loading Whisper STT Model...")
                self.stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")
                
            segments, info = self.stt_model.transcribe(temp_filename, beam_size=5)
            return " ".join([segment.text for segment in segments]).strip()
        except Exception as e:
            print(f"[ERROR] STT Failed: {e}")
            return None
        finally:
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except Exception:
                    pass

    async def _generate_audio(self, text, filename):
        communicate = edge_tts.Communicate(text, self.tts_voice)
        await communicate.save(filename)

    def synthesize(self, text):
        # We store synthesized TTS files in a separate 'db/tts_cache' directory
        # so they survive session cleanups and are reused across sessions!
        os.makedirs("db/tts_cache", exist_ok=True)
        
        # Stable cache key using stable SHA-256 hash of the text content
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        output_file = f"db/tts_cache/{text_hash}.mp3"
        
        # TTS Cache Hit: If already synthesized, skip edge-tts API network request!
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            if os.environ.get("BHARAT_LAW_VERBOSE") == "1":
                print(f"[INFO] TTS Cache Hit: {output_file}")
            return output_file
            
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
                
            if loop and loop.is_running():
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor() as executor:
                    executor.submit(asyncio.run, self._generate_audio(text, output_file)).result()
            else:
                asyncio.run(self._generate_audio(text, output_file))
                
            if os.environ.get("BHARAT_LAW_VERBOSE") == "1":
                print(f"[INFO] TTS Cache Miss. Synthesized: {output_file}")
            return output_file
        except Exception as e:
            print(f"[ERROR] TTS Failed: {e}")
            return None