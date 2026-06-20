import sounddevice as sd
import numpy as np
import socket
import threading
from collections import deque
import time
import queue
# Import here to avoid circular import
from tts_handler import (
    agent_speaking,
    agent_audio_buffer,
    agent_audio_lock
)

import wave
import os

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 160
CONTEXT_SAMPLES = 16000 * 5
VAP_HOST = '127.0.0.1'
VAP_PORT_IN = 50007

# ── Shared state (accessed by output_handler via import input_handler as ih) ──
asr_buffer = []
asr_buffer_lock = threading.Lock()

transcript_buffer = []
transcript_buffer_lock = threading.Lock()

whisper_lock = threading.Lock()

is_user_speaking = False
vad_silence_start = None   # timestamp when silence began — for latency logging

transcribe_queue = queue.Queue()

VAD_THRESHOLD = 0.02
SILENCE_TOLERANCE = 20
TRANSCRIBE_CHUNK = 16000 * 8


# ── AudioRecorder ──────────────────────────────────────────────────────────

class AudioRecorder:
    """Records user (left) and agent (right) as a time-aligned stereo WAV.

    The core problem with writing user and agent chunks separately to the
    same WAV file: WAV is a flat frame stream. Each writeframes() call appends
    frames sequentially. If user writes 160 frames then agent writes 2048
    frames, those 2048 agent frames are NOT time-aligned with the user frames
    before and after them — they appear as a block in the middle. Playback
    sounds fragmented and interleaved.

    Fix: maintain a shared time-aligned frame buffer. Both user (mic, 16kHz)
    and agent (TTS, 24kHz→resampled to 16kHz) contribute to the SAME stereo
    frame at the SAME time position. A single writer thread flushes aligned
    stereo frames to disk at regular intervals.

    Architecture:
    - _user_buf: circular buffer, filled by audio_callback at 16kHz
    - _agent_buf: circular buffer, filled by TTS worker (resampled to 16kHz)
    - _writer_thread: every 100ms, pops equal number of frames from both
      buffers, zips them into stereo frames, writes to WAV
    - If one buffer has less data than the other, silence (zeros) fills the gap
    """

    AGENT_SR = 24000   # Kokoro output sample rate
    REC_SR   = 16000   # WAV recording sample rate

    def __init__(self, session_id: str):
        os.makedirs("Logs/audio", exist_ok=True)
        filename = f"Logs/audio/audio_{session_id}.wav"

        self._wav = wave.open(filename, 'w')
        self._wav.setnchannels(2)
        self._wav.setsampwidth(2)        # 16-bit PCM
        self._wav.setframerate(self.REC_SR)

        # Separate sample queues for each channel
        self._user_buf  = []
        self._agent_buf = []
        self._buf_lock  = threading.Lock()

        self._running = True
        self._writer = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="AudioRecWriter"
        )
        self._writer.start()

        print(f"[AudioRec]  → {filename}")

    def write_user(self, chunk: np.ndarray):
        """Called from audio_callback — 160 float32 samples at 16kHz."""
        pcm = (chunk * 32767).astype(np.int16)
        with self._buf_lock:
            self._user_buf.extend(pcm.tolist())

    def write_agent(self, chunk: np.ndarray):
        """Called from TTS worker — float32 samples at 24kHz.
        Resamples to 16kHz before buffering so both channels share same SR.
        """
        from scipy.signal import resample_poly
        # 24000 * 2/3 = 16000 — exact integer ratio, no quality loss
        chunk_16k = resample_poly(chunk.astype(np.float32), up=2, down=3)
        pcm = (chunk_16k * 32767).astype(np.int16)
        with self._buf_lock:
            self._agent_buf.extend(pcm.tolist())

    def _writer_loop(self):
        """Runs every 100ms — flushes time-aligned stereo frames to WAV.
        Pops N frames from both buffers where N = min(len(user), len(agent),
        frames_in_100ms). Fills the shorter buffer with silence.
        This guarantees frame-level time alignment regardless of write timing.
        """
        FLUSH_SAMPLES = self.REC_SR // 10  # 1600 samples = 100ms

        while self._running:
            time.sleep(0.1)

            with self._buf_lock:
                n_user  = len(self._user_buf)
                n_agent = len(self._agent_buf)
                # Flush however much the slower channel has produced
                n = min(FLUSH_SAMPLES, n_user)

                if n == 0:
                    continue

                user_samples  = self._user_buf[:n]
                agent_samples = self._agent_buf[:n] if n_agent >= n \
                                else self._agent_buf + [0] * (n - n_agent)

                del self._user_buf[:n]
                del self._agent_buf[:min(n, n_agent)]

            # Interleave into stereo: L=user, R=agent
            stereo = np.zeros(n * 2, dtype=np.int16)
            stereo[0::2] = np.array(user_samples,  dtype=np.int16)
            stereo[1::2] = np.array(agent_samples[:n], dtype=np.int16)

            if self._wav._file is not None:
                self._wav.writeframes(stereo.tobytes())

    def close(self):
        self._running = False
        self._writer.join(timeout=2)
        # Flush remaining samples
        with self._buf_lock:
            n = max(len(self._user_buf), len(self._agent_buf))
            if n > 0:
                u = self._user_buf  + [0] * max(0, n - len(self._user_buf))
                a = self._agent_buf + [0] * max(0, n - len(self._agent_buf))
                stereo = np.zeros(n * 2, dtype=np.int16)
                stereo[0::2] = np.array(u[:n], dtype=np.int16)
                stereo[1::2] = np.array(a[:n], dtype=np.int16)
                if self._wav._file is not None:
                    self._wav.writeframes(stereo.tobytes())
        if self._wav._file is not None:
            self._wav.close()


# Module-level instance — set by InputHandler.__init__
audio_recorder: 'AudioRecorder | None' = None


# ── Module-level functions (accessed as ih.xxx) ────────────────────────────

def incremental_transcriber(whisper_model, running_flag):
    """Background thread — transcribes 8s audio chunks while user speaks.
    Results accumulate in transcript_buffer for get_full_transcript().
    """
    while running_flag[0]:
        try:
            audio = transcribe_queue.get(timeout=0.5)
            with whisper_lock:
                segments, _ = whisper_model.transcribe(
                    audio, language="en", vad_filter=True
                )
            text = " ".join([s.text.strip() for s in segments])
            if text.strip():
                with transcript_buffer_lock:
                    transcript_buffer.append(text)
                print(f"[Incremental] {text}")
        except queue.Empty:
            continue


def get_full_transcript(whisper_model):
    """Called when VAP fires turn end.
    Combines all incremental transcript chunks with any remaining audio.
    Returns complete transcript regardless of how long user spoke.
    """
    with asr_buffer_lock:
        remaining_audio = np.array(asr_buffer, dtype=np.float32)
        asr_buffer.clear()

    remaining_text = ""
    if len(remaining_audio) > 3200:
        with whisper_lock:
            segments, _ = whisper_model.transcribe(
                remaining_audio, language="en", vad_filter=True
            )
        remaining_text = " ".join([s.text.strip() for s in segments])

    with transcript_buffer_lock:
        parts = transcript_buffer.copy()
        transcript_buffer.clear()

    if remaining_text.strip():
        parts.append(remaining_text)

    return " ".join(parts).strip()


# ── InputHandler class ──────────────────────────────────────────────────────

class InputHandler:
    def __init__(self, whisper_model, session_id: str):
        global audio_recorder
        self.vap_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.vap_socket.connect((VAP_HOST, VAP_PORT_IN))
        print(f"Connected to VAP server on port {VAP_PORT_IN}")

        self.whisper_model = whisper_model
        self.running = True
        self.running_flag = [True]

        self.context_buffer = deque(maxlen=CONTEXT_SAMPLES)
        self.agent_buffer = deque(maxlen=CONTEXT_SAMPLES)
        self.context_buffer.extend(np.zeros(CONTEXT_SAMPLES, dtype=np.float32))
        self.agent_buffer.extend(np.zeros(CONTEXT_SAMPLES, dtype=np.float32))

        self.silence_frames = 0
        self.send_queue = deque(maxlen=50)
        self.buffer_lock = threading.Lock()

        audio_recorder = AudioRecorder(session_id)

    def audio_callback(self, indata, frames, time_info, status):
        global is_user_speaking, vad_silence_start
        chunk = indata[:, 0].astype(np.float32)

        energy = np.sqrt(np.mean(chunk**2))

        if energy > VAD_THRESHOLD:
            # User started speaking
            if not is_user_speaking:
                # Rising edge — user just started speaking
                # If agent was speaking, this is an interruption
                agent_speaking.clear()  # stop TTS immediately
            is_user_speaking = True
            self.silence_frames = 0
            vad_silence_start = None
        else:
            self.silence_frames += 1
            if self.silence_frames > SILENCE_TOLERANCE:
                if is_user_speaking:
                    # Falling edge — user just stopped speaking
                    vad_silence_start = time.time()
                is_user_speaking = False

        # Accumulate ASR buffer while user speaking
        # Push to incremental transcriber every 8 seconds
        with asr_buffer_lock:
            if is_user_speaking:
                asr_buffer.extend(chunk.tolist())
                if len(asr_buffer) >= TRANSCRIBE_CHUNK:
                    audio_to_transcribe = np.array(asr_buffer, dtype=np.float32)
                    asr_buffer.clear()
                    transcribe_queue.put(audio_to_transcribe)

        # Queue chunk for VAP sender
        with self.buffer_lock:
            self.context_buffer.extend(chunk)
            self.send_queue.append(chunk.copy())

        # Record user audio — every chunk regardless of VAD state
        if audio_recorder is not None:
            audio_recorder.write_user(chunk)

    def send_to_vap(self):
        while self.running:
            chunk_to_send = None
            with self.buffer_lock:
                if self.send_queue:
                    chunk_to_send = self.send_queue.popleft()

            if chunk_to_send is not None:
                user_chunk = chunk_to_send.astype(np.float64)
                with agent_audio_lock:

                    available = min(
                        len(agent_audio_buffer),
                        CHUNK_SAMPLES
                    )

                    samples = [
                        agent_audio_buffer.popleft()
                        for _ in range(available)
                    ]

                if available < CHUNK_SAMPLES:
                    samples.extend(
                        [0.0] * (CHUNK_SAMPLES - available)
                    )

                agent_chunk = np.array(
                    samples,
                    dtype=np.float64
                )
                interleaved = np.empty(CHUNK_SAMPLES * 2, dtype=np.float64)
                interleaved[0::2] = user_chunk
                interleaved[1::2] = agent_chunk
                try:
                    self.vap_socket.sendall(interleaved.tobytes())
                except Exception as e:
                    print(f"Send error: {e}")
            else:
                time.sleep(0.001)

    def start(self):
        # Incremental transcriber thread
        threading.Thread(
            target=incremental_transcriber,
            args=(self.whisper_model, self.running_flag),
            daemon=True,
            name="IncrementalTranscriber"
        ).start()

        # VAP sender thread
        threading.Thread(
            target=self.send_to_vap,
            daemon=True,
            name="VAPSender"
        ).start()

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK_SAMPLES,
            dtype='float32',
            callback=self.audio_callback,
            device=1
        )
        print("Microphone stream starting...")
        with self.stream:
            while self.running:
                sd.sleep(100)

    def stop(self):
        self.running = False
        self.running_flag[0] = False
        if audio_recorder is not None:
            audio_recorder.close()