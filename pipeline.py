import onnxruntime as ort
ort.preload_dlls()  # handles all DLL loading automatically
import threading
import time
import os
from datetime import datetime
from faster_whisper import WhisperModel
from input_handler import InputHandler
from output_handler import OutputHandler, SYSTEM_NAME
from tts_handler import agent_speaking


def check_ollama():
    try:
        import ollama
        ollama.list()
        print("Ollama running ✓")
        return True
    except Exception:
        print("ERROR: Ollama not running. Run: ollama serve")
        return False


def main():
    # Session ID shared across all log files — ties audio WAV to CSV rows
    session_id = f"{SYSTEM_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create log directories — fail early rather than at first log write
    for d in ["Logs/eventLogger", "Logs/turnLogger", "Logs/audio"]:
        os.makedirs(d, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"VAP Turn-Taking Pipeline — System: {SYSTEM_NAME}")
    print(f"Session: {session_id}")

    if not check_ollama():
        return

    # Load Whisper once — shared between InputHandler and OutputHandler
    print("Loading Whisper...")
    whisper = WhisperModel("tiny.en", device="cuda", compute_type="float16")
    print("Whisper loaded ✓")

    try:
        input_handler = InputHandler(whisper, session_id)
    except ConnectionRefusedError:
        print("\nERROR: VAP server not running.")
        print("Start it first:")
        print("  python rvap/vap_main/vap_main.py "
              "--model asset/vap/vap_state_dict_eng_10hz_5000msec_MC.pt "
              "--vap_process_rate 10 --context_len_sec 5\n")
        return
    
    output_handler = OutputHandler(input_handler, whisper)
    
    t1 = threading.Thread(
        target=input_handler.start,
        daemon=True,
        name="MicCapture"
    )
    t3 = threading.Thread(
        target=output_handler.start,
        daemon=True,
        name="TurnProcessor"
    )
    

    t1.start()
    time.sleep(2)  # let mic stabilise before connecting VAP output
    t3.start()


    print("\nPipeline running. Speak naturally.\n")
    print("NOTE: Wear headphones to avoid TTS feedback into microphone.\n")  


    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
        input_handler.stop()
        output_handler.running = False
        print(f"Logs saved — session: {session_id}")
        print("  Logs/eventLogger/  Logs/turnLogger/  Logs/audio/")
    

if __name__ == "__main__":
    main()