import os
import asyncio
import edge_tts
# REMOVED: from faster_whisper import WhisperModel (Moved inside class)

class VoiceHandler:
    def __init__(self):
        print("[INFO] Initializing Voice Models...")
        
        # --- LAZY IMPORT ---
        # This prevents the app from hanging just by importing this file
        from faster_whisper import WhisperModel
        
        # Initialize Model
        self.stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        self.tts_voice = "en-IN-NeerjaNeural" 

    def transcribe(self, audio_bytes):
        temp_filename = "temp_input.wav"
        with open(temp_filename, "wb") as f:
            f.write(audio_bytes)
        try:
            segments, info = self.stt_model.transcribe(temp_filename, beam_size=5)
            return " ".join([segment.text for segment in segments]).strip()
        except Exception as e:
            print(f"[ERROR] STT Failed: {e}")
            return None

    async def _generate_audio(self, text, filename):
        communicate = edge_tts.Communicate(text, self.tts_voice)
        await communicate.save(filename)

    def synthesize(self, text):
        output_file = "response_audio.mp3"
        try:
            asyncio.run(self._generate_audio(text, output_file))
            return output_file
        except Exception as e:
            print(f"[ERROR] TTS Failed: {e}")
            return None