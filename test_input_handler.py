# test_input.py
from input_handler import InputHandler
import time
import ollama

handler = InputHandler()
import threading
t = threading.Thread(target=handler.start, daemon=True)
t.start()

print("Speak for 5 seconds...")
time.sleep(5)
audio = handler.get_asr_audio()
print(f"Captured {len(audio)} samples = {len(audio)/16000:.1f} seconds of audio")
handler.stop()