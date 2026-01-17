import os
import asyncio
from faster_whisper import WhisperModel
import edge_tts

class VoiceHandler:
    def __init__(self):
        print("[INFO] Loading Voice Models... (This might take a moment on first run)")
        
        # 1. Initialize STT (Speech to Text)
        # using 'tiny' for speed. Change to 'base' or 'small' for better accuracy.
        # compute_type="int8" makes it fast even on CPU.
        self.stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        
        # 2. Configure TTS (Text to Speech)
        # "en-IN-NeerjaNeural" is a high-quality Indian female voice
        self.tts_voice = "en-IN-NeerjaNeural" 

    def transcribe(self, audio_bytes):
        """
        Convert Audio to Text using Local Whisper
        """
        temp_filename = "temp_input.wav"
        
        # Write bytes to file
        with open(temp_filename, "wb") as f:
            f.write(audio_bytes)
        
        try:
            # Transcribe
            segments, info = self.stt_model.transcribe(temp_filename, beam_size=5)
            
            # Combine segments into one string
            transcribed_text = " ".join([segment.text for segment in segments])
            return transcribed_text.strip()
            
        except Exception as e:
            print(f"[ERROR] STT Failed: {e}")
            return None

    def synthesize(self, text, output_file="output_speech.mp3"):
        """
        Convert Text to Audio using Edge TTS (Free)
        """
        try:
            # edge-tts is asynchronous, so we use asyncio.run
            asyncio.run(self._save_audio(text, output_file))
            return output_file
        except Exception as e:
            print(f"[ERROR] TTS Failed: {e}")
            return None

    async def _save_audio(self, text, output_file):
        """Helper for async TTS"""
        communicate = edge_tts.Communicate(text, self.tts_voice)
        await communicate.save(output_file)