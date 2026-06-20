"""
TTS Handler — Kokoro-based streaming TTS for voice AI pipeline.
Plays agent responses sentence-by-sentence as LLM generates.
Supports interruption: if user starts speaking, TTS stops immediately.
"""
import onnxruntime as ort
ort.preload_dlls()  # handles all DLL loading automatically
import threading
import time
import numpy as np
import input_handler as ih
from kokoro_onnx import Kokoro
import sounddevice as sd
from onnxruntime import InferenceSession
from scipy.signal import resample_poly

# agent_speaking flag — shared with input_handler for interruption detection
agent_speaking = threading.Event()

from collections import deque

# Agent audio forwarded to VAP
agent_audio_buffer = deque(maxlen=16000 * 5)
agent_audio_lock = threading.Lock()

# TTS audio output stream — lazy initialized
_output_stream = None
_stream_lock = threading.Lock()

def _get_output_stream(sample_rate=24000):
    global _output_stream
    with _stream_lock:
        if _output_stream is None:
            _output_stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype='float32'
            )
            _output_stream.start()
    return _output_stream


class TTSHandler:
    def __init__(self):
        print("Loading Kokoro TTS...")
        try:
            
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 4

            inf_sess = InferenceSession(
                "kokoro-v1.0.onnx",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                sess_options=sess_options
            )
            if "CUDAExecutionProvider" in inf_sess.get_providers():
                print("Using CUDAExecutionProvider for TTS")
            self.kokoro = Kokoro.from_session(inf_sess, "voices-v1.0.bin")
            self.voice = "af_sarah"
            self.sample_rate = 24000
            print("Kokoro TTS loaded ✓")
            self.available = True
        except Exception as e:
            print(f"[TTS] Kokoro not available: {e}")
            print("[TTS] Running in text-only mode")
            self.available = False


        self.tts_start_time = None   # time TTS first audio chunk played

        # OVERLAP FIX: single lock that serializes all speak_sentence calls.
        # Each sentence thread acquires this before playing and holds it until
        # done. The next thread blocks at acquire() until the previous finishes.
        # Only one thread writes to _output_stream at any moment — no overlap.
        self._play_lock = threading.Lock()

    def speak_sentence(self, text, on_first_audio=None):
        """
        Speak one sentence. Non-blocking — runs in caller's thread.
        Stops immediately if agent_speaking is cleared (user interruption).
        on_first_audio: callback fired when first audio chunk plays.
        Returns: 'completed', 'interrupted', or 'skipped'
        """
        if not text.strip():
            return 'skipped'

        if not self.available:
            # Text-only mode — just print, simulate timing
            print(f"\n[TTS-TEXT] {text}")
            if on_first_audio:
                on_first_audio()
            return 'completed'

        try:

            # Generate audio — outside lock so generation can overlap with
            # the previous sentence still playing. Only playback is serialized.

            # OVERLAP FIX: acquire lock before writing to stream.
            # Blocks here until previous sentence finishes playing.
            # Releases automatically when this sentence finishes or is interrupted.
            with self._play_lock:
                # Check interruption again after waiting for lock —
                # may have been interrupted while waiting
                if not agent_speaking.is_set():
                    return 'interrupted'
                audio, sr = self.kokoro.create(
                    text, voice=self.voice, speed=1.0, lang="en-us"
                )

                audio_16k = resample_poly(
                    audio.astype(np.float32),
                    up=2,
                    down=3
                )

                with agent_audio_lock:
                    agent_audio_buffer.extend(audio_16k.tolist())
                audio = audio.astype(np.float32)

                stream = _get_output_stream(sr)

                CHUNK = 512
                first_chunk = True

                for i in range(0, len(audio), CHUNK):
                    if not agent_speaking.is_set():
                        return 'interrupted'

                    chunk = audio[i:i+CHUNK]
                    stream.write(chunk)
                    ih.audio_recorder.write_agent(chunk)

                    if first_chunk:
                        if on_first_audio:
                            on_first_audio()
                        first_chunk = False

            return 'completed'

        except Exception as e:
            print(f"[TTS] Error: {e}")
            return 'error'

    def stop(self):
        """Signal TTS to stop — called when user interrupts."""
        agent_speaking.clear()