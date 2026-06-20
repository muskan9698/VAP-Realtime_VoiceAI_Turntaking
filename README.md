# VAP-Realtime for Voice AI Turn-Taking

This repo is a fork of [VAP-Realtime](https://github.com/inokoj/VAP-Realtime) with an added voice
AI pipeline integration — wiring VAP's real-time turn-taking prediction into a full
ASR → LLM → TTS voice agent.

> Part of a larger project: [Turn-Taking in Voice AI Calls](https://github.com/muskan9698/Voice-AI-Turn-Taking-Implementation)
> See the [DualTurn side of this comparison here](https://github.com/muskan9698/Dualturn_VoiceAI_Turntaking).

## What is VAP-Realtime

[VAP-Realtime](https://github.com/inokoj/VAP-Realtime) (Voice Activity Projection), based on [Paper (arXiv)](https://arxiv.org/pdf/2401.04868) is a real-time
turn-taking prediction model. It checks audio frames every 10ms and outputs two probabilities,
`p_now` and `p_future`, predicting whether the current speaker is about to yield the floor.

## What's added in this fork

ASR-LLM-TTS + turn taking predictions using VAP_Realtime

The original VAP-Realtime model code and core VAP server remain unchanged.

## Setup

```bash
# Create and activate a virtual environment (Python 3.11)
git clone https://github.com/muskan9698/VAP-Realtime_VoiceAI_Turntaking.git
cd VAP-Realtime_VoiceAI_Turntaking
```

Install dependencies — **GPU users**, use `requirements-gpu.txt`; otherwise use `requirements.txt`:

```bash
# GPU
pip install -r requirements-gpu.txt

# CPU only
pip install -r requirements.txt
```

```bash
pip install numpy==1.26.4 sounddevice scipy

# PyTorch with CUDA 12.1 support (skip/swap for CPU-only torch if not using GPU)
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Additional dependencies for audio I/O and visualization
pip install einops==0.7.0 soundfile==0.12.1 pygame pydub==0.25.1 pyaudio matplotlib==3.7.5 seaborn==0.13.2 fastapi==0.111.0

# TTS (Kokoro)
pip install kokoro
pip install kokoro-onnx
curl -L -o kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

# ASR
pip install faster-whisper
```

**LLM:** install [Ollama](https://ollama.ai), then pull a model:
```bash
ollama pull phi3:mini
# (or, if not on PATH: "C:\Users\<you>\AppData\Local\Programs\Ollama\ollama.exe" pull phi3:mini)
```

## Running (3 terminals)

**Terminal 1 — launch the VAP server:**
```bash
export PYTHONPATH=.

python rvap/vap_main/vap_main.py \
    --vap_model asset/vap/vap_state_dict_eng_10hz_5000msec_MC.pt \
    --vap_process_rate 10 \
    --context_len_sec 5 \
    --gpu
```
Drop `--gpu` if running CPU-only.

**Terminal 2 — run the pipeline (or test components individually):**
```bash
export PYTHONPATH=.
python pipeline.py               # run the full voice AI pipeline
```


## Acknowledgments

This repo builds on:

Koji Inoue, Bing'er Jiang, Erik Ekstedt, Tatsuya Kawahara, Gabriel Skantze.
*Real-time and Continuous Turn-taking Prediction Using Voice Activity Projection.*
IWSDS 2024. [arXiv:2401.04868](https://arxiv.org/abs/2401.04868)

Note: the upstream repo has since been folded into
[MaAI](https://github.com/inokoj/MaAI) and may eventually be archived. This fork is based on the
standalone VAP-Realtime version.

## License

The original DualTurn code in this repo remains under the MIT License (see `LICENSE`). New files
added in this fork are also released under the Apache2.0 License in https://github.com/muskan9698/Voice-AI-Turn-Taking-Implementation.git

---

*Voice AI pipeline integration by Muskan Rathore*
