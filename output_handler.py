import socket
import struct
import numpy as np
import threading
import time
import csv
from datetime import datetime
import ollama
import input_handler as ih
from tts_handler import TTSHandler, agent_speaking

VAP_HOST = '127.0.0.1'
VAP_PORT_OUT = 50008

# Change to 'DualTurn' for Week 2 experiment
SYSTEM_NAME = "VAP"


# ── Event Logger ──────────────────────────────────────────────────────────

class EventLogger:
    def __init__(self, filename=None):
        if filename is None:
            filename = f"Logs/eventLogger/events_{SYSTEM_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.f = open(filename, 'w', newline='', encoding="utf-8")
        self.writer = csv.writer(self.f)
        self.writer.writerow([
            'event_id', 'timestamp_ms', 'event_type',
            'p_now', 'p_future', 'silence_frames',
            'vad_state', 'agent_state', 'asr_buffer_size',
            'value', 'turn_id'
        ])
        self.event_id = 0
        self.lock = threading.Lock()
        self.session_start = time.time()
        print(f"[EventLog] → {filename}")

    def log(self, event_type, p_now=0, p_future=0, silence_frames=0,
            vad_state='', agent_state='', asr_buffer_size=0, value='', turn_id=0):
        with self.lock:
            self.event_id += 1
            ts_ms = round((time.time() - self.session_start) * 1000, 1)
            self.writer.writerow([
                self.event_id, ts_ms, event_type,
                round(p_now, 4), round(p_future, 4), silence_frames,
                vad_state, agent_state, asr_buffer_size,
                value, turn_id
            ])
            self.f.flush()


# ── Turn Logger ───────────────────────────────────────────────────────────

class TurnLogger:
    def __init__(self, filename=None):
        if filename is None:
            filename = f"Logs/turnLogger/turns_{SYSTEM_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.f = open(filename, 'w', newline='', encoding="utf-8")
        self.writer = csv.writer(self.f)
        self.writer.writerow([
            'turn_id', 'system', 'timestamp',
            'user_turn_duration_sec',
            'user_transcript', 'user_word_count',
            'silence_frames_at_trigger',
            'vad_silence_to_trigger_ms',
            'p_now_at_trigger', 'p_future_at_trigger',
            'trigger_to_transcript_ms',
            'transcript_to_first_token_ms',
            'first_token_to_first_audio_ms',
            'total_time_to_first_audio_ms',
            'agent_response_text',
            'agent_response_duration_sec',
            'in_natural_window',
            'false_trigger',
            'tts_completed',
            'user_interrupted_agent',
            'agent_stop_latency_ms',
            'agent_stopped_correctly',
            'user_backchannel_during_agent',
            'agent_continued_correctly',
            'agent_backchannel_produced',
            'backchannel_text',
            'notes'
        ])
        self.turn_id = 0
        self.lock = threading.Lock()
        print(f"[TurnLog]  → {filename}")

    def log(self, data: dict):
        with self.lock:
            self.turn_id += 1
            self.writer.writerow([
                self.turn_id, SYSTEM_NAME, datetime.now().isoformat(),
                data.get('user_turn_duration_sec', ''),
                data.get('user_transcript', ''),
                data.get('user_word_count', ''),
                data.get('silence_frames_at_trigger', ''),
                data.get('vad_silence_to_trigger_ms', ''),
                data.get('p_now_at_trigger', ''),
                data.get('p_future_at_trigger', ''),
                data.get('trigger_to_transcript_ms', ''),
                data.get('transcript_to_first_token_ms', ''),
                data.get('first_token_to_first_audio_ms', ''),
                data.get('total_time_to_first_audio_ms', ''),
                data.get('agent_response_text', ''),
                data.get('agent_response_duration_sec', ''),
                data.get('in_natural_window', ''),
                data.get('false_trigger', ''),
                data.get('tts_completed', ''),
                data.get('user_interrupted_agent', 0),
                data.get('agent_stop_latency_ms', ''),
                data.get('agent_stopped_correctly', ''),
                data.get('user_backchannel_during_agent', 0),
                data.get('agent_continued_correctly', ''),
                data.get('agent_backchannel_produced', 0),
                data.get('backchannel_text', ''),
                data.get('notes', '')
            ])
            self.f.flush()
            return self.turn_id


# ── OutputHandler ─────────────────────────────────────────────────────────

class OutputHandler:
    def __init__(self, input_handler, whisper_model):
        self.input_handler = input_handler
        self.whisper = whisper_model
        self.vap_socket = None
        self.running = True
        self.processing = False

        self.tts = TTSHandler()
        self.event_log = EventLogger()
        self.turn_log = TurnLogger()

        self.current_turn_id = 0
        self.interrupt_start_time = None
        self.tts_start_time = None
        self.user_interrupted_during_tts = False
        self.silence_start_time = None

        # For tracking first audio time across threads
        self._t_first_audio = None
        self._tts_start_wall = None
        self._first_audio_lock = threading.Lock()
        print(f"{'='*50}\n")
        print("Alisha: Hi! I am Alisha, your personal therapist at Carewell. How can I help you?")
        print(f"{'='*50}\n")

        # Speak intro — OutputHandler and TTSHandler now exist
        INTRO = "Hi! I am Alisha, your personal therapist at Carewell. How can I help you?"
        agent_speaking.set()
        self.tts.speak_sentence(INTRO,None)
        agent_speaking.clear()

    def listen_for_turns(self):
        print("[OutputHandler] Started listening on port 50008")

        silence_frames = 0
        SILENCE_FRAMES_REQUIRED = 8  # 8 × ~100ms = ~0.8s — unchanged
        prev_vad_state = False

        # CHANGE 3a: p_now accumulator for averaging during silence window
        # Paper method: "averaging p_now over time during mutual silence;
        # higher participant value = prediction result" (Inoue et al. 2024)
        p_now_accumulator = []
        p_future_accumulator = []

        # CHANGE 3b: Guard against idle startup triggers
        # At startup, p_now idles at ~0.6 which would pass avg > 0.5 threshold.
        # Only start evaluating turns after user has spoken at least once.
        user_has_spoken = False

        while self.running:
            try:
                # ── Read 4-byte length prefix — UNCHANGED ────────────────
                size_data = b''
                while len(size_data) < 4:
                    chunk = self.vap_socket.recv(4 - len(size_data))
                    if not chunk:
                        print("[OutputHandler] VAP socket closed")
                        return
                    size_data += chunk

                packet_size = int.from_bytes(size_data, 'little')

                # ── Read full packet — UNCHANGED ─────────────────────────
                data = b''
                while len(data) < packet_size:
                    chunk = self.vap_socket.recv(packet_size - len(data))
                    if not chunk:
                        return
                    data += chunk

                # ── Parse VAP packet — UNCHANGED ─────────────────────────
                idx = 0
                t = struct.unpack('<d', data[idx:idx+8])[0]
                idx += 8
                len_x1 = struct.unpack('<I', data[idx:idx+4])[0]
                idx += 4 + 8 * len_x1
                len_x2 = struct.unpack('<I', data[idx:idx+4])[0]
                idx += 4 + 8 * len_x2
                len_pnow = struct.unpack('<I', data[idx:idx+4])[0]
                idx += 4
                p_now = [struct.unpack('<d', data[idx+i*8:idx+i*8+8])[0]
                         for i in range(len_pnow)]
                idx += 8 * len_pnow
                len_pfuture = struct.unpack('<I', data[idx:idx+4])[0]
                idx += 4
                p_future = [struct.unpack('<d', data[idx+i*8:idx+i*8+8])[0]
                            for i in range(len_pfuture)]
                idx += 8 * len_pfuture

                # CHANGE 1: Parse VAP's own neural VAD output
                # vap_main.py: result_vad = [vad1, vad2]
                # vad1 = va_classifier(o1) where o1 = encoder of x1 (user)
                # vad2 = va_classifier(o2) where o2 = encoder of x2 (agent)
                # We send: interleaved[0::2] = user_chunk (x1), interleaved[1::2] = agent (x2)
                # Therefore: vad[0] = user neural VAD, vad[1] = agent neural VAD
                # NOTE: vad index assignment is confirmed by source + your send_to_vap setup.
                # If behavior is inverted in testing, swap vad[0]/vad[1] below.
                len_vad = struct.unpack('<I', data[idx:idx+4])[0]
                idx += 4
                vad_values = [struct.unpack('<d', data[idx+i*8:idx+i*8+8])[0]
                              for i in range(len_vad)]
                vad_user_neural = vad_values[0] if len(vad_values) > 0 else 0.5
                # vad_agent_neural = vad_values[1] if len(vad_values) > 1 else 0.5

                p_now_agent = p_now[0] if p_now else 0.0
                p_future_agent = p_future[0] if p_future else 0.0

                # CHANGE 2: Use VAP's neural VAD for silence tracking
                # Energy VAD (ih.is_user_speaking) stays in audio_callback for
                # interruption detection — must remain fast and real-time.
                # For turn trigger decisions, neural VAD is more accurate:
                # it's trained on speech patterns, not just energy thresholds.
                # We require BOTH to agree: neural says silent AND energy says silent.
                # This makes false triggers from noise harder — dual confirmation.
                vad_state = (vad_user_neural < 0.5) is False  # True = user speaking
                # Dual check: if either says speaking, treat as speaking
                vad_state = vad_state or ih.is_user_speaking
                agent_state = agent_speaking.is_set()

                # ── VAD transition logging (NEW — additive only) ──────────
                if vad_state != prev_vad_state:
                    if vad_state:
                        # User started speaking
                        self.event_log.log(
                            'VAD_USER_START', p_now_agent, p_future_agent,
                            silence_frames, 'speaking',
                            'speaking' if agent_state else 'silent',
                            len(ih.asr_buffer),
                            turn_id=self.current_turn_id
                        )
                        if agent_state:
                            # Interruption — user spoke while agent was speaking
                            self.interrupt_start_time = time.time()
                            self.user_interrupted_during_tts = True
                            self.event_log.log(
                                'USER_INTERRUPTION', p_now_agent, p_future_agent,
                                silence_frames, 'speaking', 'speaking',
                                len(ih.asr_buffer),
                                turn_id=self.current_turn_id
                            )
                    else:
                        # User stopped speaking
                        self.silence_start_time = time.time()
                        self.event_log.log(
                            'VAD_USER_STOP', p_now_agent, p_future_agent,
                            silence_frames, 'silent',
                            'speaking' if agent_state else 'silent',
                            len(ih.asr_buffer),
                            turn_id=self.current_turn_id
                        )
                prev_vad_state = vad_state

                # ── Silence counter — updated to use dual VAD + accumulate p_now ──
                if not vad_state:
                    silence_frames += 1
                    p_now_accumulator.append(p_now_agent)  # accumulate during silence
                    p_future_accumulator.append(p_future_agent)
                else:
                    silence_frames = 0
                    p_now_accumulator = []                 # reset on any speech
                    p_future_accumulator = []              # reset future probabilities
                    if len(ih.asr_buffer) > 0 or len(ih.transcript_buffer) > 0:
                        user_has_spoken = True             # user has produced audio

                # ── Debug print — updated to show neural VAD ──────────────
                if not self.processing:
                    avg_p_n = round(sum(p_now_accumulator)/len(p_now_accumulator), 3) if p_now_accumulator else 0.0
                    avg_p_f = round(sum(p_future_accumulator)/len(p_future_accumulator), 3) if p_future_accumulator else 0.0

                    print(f"p_now={p_now_agent:.3f} | p_future={p_future_agent:.3f} | "
                          f"buf={len(ih.asr_buffer)} | "
                          f"vad_e={'spk' if ih.is_user_speaking else 'sil'} | "
                          f"vad_n={'spk' if vad_user_neural >= 0.5 else 'sil'}({vad_user_neural:.2f}) | "
                          f"agent={'speaking' if agent_state else 'silent'} | "
                          f"sf={silence_frames}/{SILENCE_FRAMES_REQUIRED} | "
                          f"avg_p_n={avg_p_n} |"
                          f"avg_p_f={avg_p_f}")

                # ── Trigger condition — VAP paper method ─────────────────
                # Paper: "averaging p_now over time during mutual silence;
                # higher value = agent should take floor" (Inoue et al. 2024)
                # avg_p_now > 0.5 means VAP predicts agent should speak
                # (since p_now[agent] + p_now[user] ≈ 1.0, > 0.5 = agent wins)
                has_audio = len(ih.asr_buffer) > 3200
                has_transcript = len(ih.transcript_buffer) > 0
                

                if (silence_frames >= SILENCE_FRAMES_REQUIRED and
                    not self.processing and
                    user_has_spoken and                    # guard: ignore idle startup
                    (has_audio or has_transcript)):

                    # Compute avg p_now over silence window — paper's method
                    window = p_now_accumulator[-SILENCE_FRAMES_REQUIRED:]
                    avg_p_now = sum(window) / len(window) if window else 0.0
                    window = p_future_accumulator[-SILENCE_FRAMES_REQUIRED:]
                    avg_p_future = sum(window) / len(window) if window else 0.0


                    if avg_p_now > 0.5 and avg_p_future > 0.5:
                        # SHIFT: VAP + VAD both agree — agent should speak
                        actual_silence_frames = silence_frames
                        vad_silence_ms = round(actual_silence_frames * 100, 1)
                        trigger_time = time.time()

                        self.event_log.log(
                            'VAP_TRIGGER', p_now_agent, p_future_agent,
                            actual_silence_frames, 'silent',
                            'speaking' if agent_state else 'silent',
                            len(ih.asr_buffer),
                            value=f'sf={actual_silence_frames},avg_p_now={avg_p_now:.3f},avg_p_future={avg_p_future:.3f}',
                            turn_id=self.current_turn_id + 1
                        )

                        silence_frames = 0
                        p_now_accumulator = []
                        self.processing = True
                        self.user_interrupted_during_tts = False

                        print(f"\n>>> TURN END — p_now={p_now_agent:.3f} "
                              f"p_future={p_future_agent:.3f} "
                              f"avg_p_now={avg_p_now:.3f} "
                              f"avg_p_future={avg_p_future:.3f} "
                              f"buf={len(ih.asr_buffer)} "
                              f"chunks={len(ih.transcript_buffer)}")

                        threading.Thread(
                            target=self.process_turn,
                            args=(trigger_time, p_now_agent, p_future_agent,
                                  actual_silence_frames, vad_silence_ms),
                            daemon=True
                        ).start()
                    else:
                        # HOLD: VAD says silent but VAP says user will continue
                        # Log for research — this is the key VAP advantage over pure VAD
                        self.event_log.log(
                            'VAP_HOLD', p_now_agent, p_future_agent,
                            silence_frames, 'silent',
                            'speaking' if agent_state else 'silent',
                            len(ih.asr_buffer),
                            value=f'avg_p_now={avg_p_now:.3f},avg_p_future={avg_p_future:.3f}_held',
                            turn_id=self.current_turn_id
                        )
                        print(f"[HOLD] sf={silence_frames} avg_p_now={avg_p_now:.3f} < 0.5 or avg_p_future={avg_p_future:.3f} < 0.5 — VAP predicts user continues")

            except Exception as e:
                print(f"VAP receive error: {e}")
                import traceback
                traceback.print_exc()
                break

    def _speak_sentence_thread(self, sentence, trigger_time, is_first):
        """Runs TTS for one sentence in background thread.
        FIX: defined as proper method, not nested function inside loop.
        Records first audio time on first sentence only.
        """
        def on_first_audio():
            with self._first_audio_lock:
                if self._t_first_audio is None:
                    self._tts_start_wall = time.time()
                    self._t_first_audio = (time.time() - trigger_time) * 1000

        result = self.tts.speak_sentence(
            sentence,
            on_first_audio=on_first_audio if is_first else None
        )
        return result

    def process_turn(self, trigger_time, p_now, p_future,
                     actual_silence_frames, vad_silence_ms):

        turn_data = {
            'p_now_at_trigger': round(p_now, 4),
            'p_future_at_trigger': round(p_future, 4),
            'silence_frames_at_trigger': actual_silence_frames,
            'vad_silence_to_trigger_ms': vad_silence_ms,
        }

        # ── Step 1: Transcribe — UNCHANGED ───────────────────────────────
        t0 = time.time()
        transcript = ih.get_full_transcript(self.whisper)
        t_whisper = (time.time() - t0) * 1000

        word_count = len(transcript.split()) if transcript else 0
        turn_data['trigger_to_transcript_ms'] = round(t_whisper, 1)
        turn_data['user_transcript'] = transcript
        turn_data['user_word_count'] = word_count

        print(f"[Turn] Transcript ({word_count}w) in {t_whisper:.0f}ms")

        if not transcript.strip():
            self.event_log.log('FALSE_TRIGGER', p_now, p_future,
                               value='empty transcript',
                               turn_id=self.current_turn_id + 1)
            turn_data['false_trigger'] = 1
            self.turn_log.log(turn_data)
            self.processing = False
            return

        turn_data['false_trigger'] = 1 if word_count < 3 else 0

        # Truncate to last 150 words for LLM speed — UNCHANGED
        words = transcript.split()
        if len(words) > 150:
            transcript = " ".join(words[-150:])

        print(f"\nUser: {transcript}")
        print(f"[Whisper: {t_whisper:.0f}ms]")

        # ── Step 2: LLM streaming → sentence-by-sentence TTS ─────────────
        t1 = time.time()
        t_first_token = None
        full_response = ""
        sentence_buffer = ""
        tts_threads = []

        # Reset first audio tracker
        with self._first_audio_lock:
            self._t_first_audio = None
            self._tts_start_wall = None

        agent_speaking.set()
        self.event_log.log('TTS_START', p_now, p_future,
                           turn_id=self.current_turn_id + 1)

        print("Alisha: ", end="", flush=True)

        for chunk in ollama.chat(
            model="phi3:mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Alisha, a warm therapist at Carewell. "
                        "Reply warmly in 1-2 sentence only. No more."
                    )
                },
                {"role": "user", "content": transcript}
            ],
            stream=True
        ):
            token = chunk['message']['content']
            print(token, end="", flush=True)
            full_response += token
            sentence_buffer += token

            if t_first_token is None:
                t_first_token = (time.time() - t1) * 1000

            # FIX: TTS runs in background thread so LLM streaming continues
            # unblocked. is_first captured via len(tts_threads) at call time.
            if any(sentence_buffer.rstrip().endswith(p)
                   for p in ['.', '!', '?', '...']):
                sentence = sentence_buffer.strip()
                sentence_buffer = ""
                if sentence and agent_speaking.is_set():
                    is_first = len(tts_threads) == 0
                    t = threading.Thread(
                        target=self._speak_sentence_thread,
                        args=(sentence, trigger_time, is_first),
                        daemon=True
                    )
                    t.start()
                    tts_threads.append(t)

        # Speak any remaining text
        if sentence_buffer.strip() and agent_speaking.is_set():
            is_first = len(tts_threads) == 0
            t = threading.Thread(
                target=self._speak_sentence_thread,
                args=(sentence_buffer.strip(), trigger_time, is_first),
                daemon=True
            )
            t.start()
            tts_threads.append(t)

        print()

        # Wait for all TTS threads to finish before clearing agent_speaking
        for t in tts_threads:
            t.join()

        agent_speaking.clear()

        self.event_log.log('TTS_STOP', p_now, p_future,
                           turn_id=self.current_turn_id + 1)

        # ── Step 3: Compute and log metrics ──────────────────────────────
        t_first_token = t_first_token if t_first_token else 0

        with self._first_audio_lock:
            t_first_audio = self._t_first_audio

        total_first_audio = t_first_audio if t_first_audio else (
            (time.time() - trigger_time) * 1000
        )

        in_natural = 1 if 200 <= total_first_audio <= 800 else 0
        tts_status = 'completed' if tts_threads else 'text-only'
        if not self.tts.available:
            tts_status = 'text-only'

        # Interruption metrics
        interrupt_stop_ms = ''
        agent_stopped_ok = ''
        if (self.user_interrupted_during_tts and
                self.interrupt_start_time and self._tts_start_wall):
            interrupt_stop_ms = round(
                (time.time() - self.interrupt_start_time) * 1000, 1
            )
            agent_stopped_ok = 1 if interrupt_stop_ms < 500 else 0

        turn_data.update({
            'transcript_to_first_token_ms': round(t_first_token, 1),
            'first_token_to_first_audio_ms': round(
                total_first_audio - t_whisper - t_first_token, 1
            ) if t_first_audio else '',
            'total_time_to_first_audio_ms': round(total_first_audio, 1),
            'agent_response_text': full_response,
            'in_natural_window': in_natural,
            'tts_completed': tts_status,
            'user_interrupted_agent': 1 if self.user_interrupted_during_tts else 0,
            'agent_stop_latency_ms': interrupt_stop_ms,
            'agent_stopped_correctly': agent_stopped_ok,
            'agent_backchannel_produced': 0,  # VAP cannot predict BC
        })

        self.current_turn_id = self.turn_log.log(turn_data)

        t_total = (time.time() - trigger_time) * 1000
        print(f"[Whisper {t_whisper:.0f}ms | "
              f"LLM first {t_first_token:.0f}ms | "
              f"First audio {total_first_audio:.0f}ms | "
              f"Total {t_total:.0f}ms | "
              f"Natural={'✓' if in_natural else '✗'}]\n")

        self.processing = False
        self.user_interrupted_during_tts = False

    def start(self):
        # ── UNCHANGED from working version ────────────────────────────────
        print("Connecting to VAP output port...")
        self.vap_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.vap_socket.connect((VAP_HOST, VAP_PORT_OUT))
        print(f"Connected to VAP output on port {VAP_PORT_OUT}")
        self.listen_for_turns()