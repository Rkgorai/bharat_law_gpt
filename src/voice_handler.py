import os
import asyncio
import subprocess
import shutil
from faster_whisper import WhisperModel
import edge_tts

class VoiceHandler:
    def __init__(self):
        print("[INFO] Loading Voice Models...")
        self.stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        self.tts_voice = "en-IN-NeerjaNeural" 
        self.current_process = None

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

    def stream_audio(self, text):
        try:
            asyncio.run(self._stream_to_speakers(text))
            return True
        except Exception as e:
            print(f"[ERROR] Streaming Failed: {e}")
            return False

    def stop_audio(self):
        """Kills the active audio process"""
        print("[INFO] Stopping Audio...")
        if self.current_process:
            try:
                self.current_process.terminate()
                self.current_process.wait(timeout=0.2)
            except Exception:
                pass
            self.current_process = None
        # Force kill any lingering mpv instances
        os.system("pkill mpv")

    def is_playing(self):
        """Checks if audio is currently playing"""
        if self.current_process:
            # poll() returns None if process is still running
            if self.current_process.poll() is None:
                return True
        return False

    async def _stream_to_speakers(self, text):
        if not shutil.which("mpv"):
            raise EnvironmentError("mpv not found.")

        communicate = edge_tts.Communicate(text, self.tts_voice)
        
        self.current_process = subprocess.Popen(
            ["mpv", "--no-cache", "--no-terminal", "--", "-"],
            stdin=subprocess.PIPE
        )

        try:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    if self.current_process.stdin:
                        self.current_process.stdin.write(chunk["data"])
                        self.current_process.stdin.flush()
        except Exception:
            pass
        finally:
            if self.current_process and self.current_process.stdin:
                self.current_process.stdin.close()
            if self.current_process:
                self.current_process.wait()
            # Mark process as done so is_playing() returns False
            self.current_process = None