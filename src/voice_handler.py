import os
import asyncio
import edge_tts
from faster_whisper import WhisperModel

class VoiceHandler:
    def __init__(self):
        print("[INFO] Loading Voice Models...")
        self.stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        self.tts_voice = "en-IN-NeerjaNeural" 

    def transcribe(self, audio_bytes):
        """Convert Audio to Text"""
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
        """Helper to generate audio file async"""
        communicate = edge_tts.Communicate(text, self.tts_voice)
        await communicate.save(filename)

    def synthesize(self, text):
        """
        Generates an audio file and returns the path.
        Compatible with Web Deployment.
        """
        output_file = "response_audio.mp3"
        try:
            # Run the async function in a blocking way for Streamlit
            asyncio.run(self._generate_audio(text, output_file))
            return output_file
        except Exception as e:
            print(f"[ERROR] TTS Failed: {e}")
            return None