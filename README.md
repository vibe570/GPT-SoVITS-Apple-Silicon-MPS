<div align="center">

<h1>GPT-SoVITS · Apple Silicon MPS Optimized</h1>
An Apple Silicon (M1/M2/M3/M4/M5) optimized fork of GPT-SoVITS — end-to-end MPS fp32 inference, training, and UVR5 acceleration. No CUDA required.<br><br>

[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge&logo=python)](https://www.python.org)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge&logo=opensourceinitiative)](https://github.com/vibe570/GPT-SoVITS-Apple-Silicon-MPS/blob/main/LICENSE)

**English** | [**中文简体**](./docs/cn/README.md)

</div>

---

## 🍎 What does this fork do?

Based on the upstream [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) main branch, this fork adds **end-to-end MPS adaptation** for Apple Silicon. The upstream mostly leaves macOS users on CPU; this fork wires MPS all the way through — inference, training, data preprocessing, UVR5, and FunASR. One command after clone: `install.sh --device MPS`. No code changes needed, no CUDA.

### How it differs from upstream

| Capability | Upstream | This fork |
|------------|----------|-----------|
| macOS TTS inference | CPU only | **Auto-detect MPS, fp32 inference** (v2Pro/v3 weights crash with fp16 on MPS, so fp32 is enforced) |
| macOS training (s1/s2) | Not specifically adapted | **s1/s2/s2_v3_lora full-pipeline MPS fp32 training** |
| Data preprocessing (hubert/sv/semantic/get-text) | Not specifically adapted | **MPS auto-fallback, zero code changes** |
| UVR5 vocal separation | Not specifically adapted | **MPS fp32 weights + fp16 autocast compute** — avoids matrix-mult crashes while preserving accuracy |
| FunASR ASR | Not specifically adapted | **MPS device auto-selected** |
| `PYTORCH_ENABLE_MPS_FALLBACK` | Manual `export` required | **Auto `setdefault`** — unimplemented ops fall back to CPU |

Core changed files: `GPT_SoVITS/TTS_infer_pack/TTS.py`, `config.py`, `GPT_SoVITS/inference_webui.py`, `s1_train.py` / `s2_train.py` / `s2_train_v3_lora.py`, `prepare_datasets/*.py`, `tools/asr/funasr_asr.py`.

---

## ⚡ Benchmarks (Apple M5 / 32 GB RAM)

### 1. TTS inference (v2ProPlus weights)

| Config | Time for 27 chars | Notes |
|--------|-------------------|-------|
| Upstream stock MPS fp32 | 7.08 s | — |
| **This fork, MPS fp32** | **6.79 s** | ~4% faster |
| CPU fallback | ~60+ s | Reference |

On MPS, `v2Pro` / `v3` / `v3_lora` weights **crash immediately with fp16**. This fork forces `is_half=False` automatically — no config edit needed.

### 2. UVR5 vocal separation (bs_roformer)

| Config | Time per chunk | SNR (end-to-end) |
|--------|----------------|------------------|
| Upstream CPU | 31 – 61 s/chunk | Baseline |
| **This fork, MPS** | **9.6 s/chunk** | **68.7 dB** (perceptually lossless) |

That's **3–6× faster**. The "fp32 weights + fp16 autocast compute" strategy eliminates MPS assertion crashes on matrix multiplication while keeping accuracy loss negligible.

### 3. Long-form TTS (same project, measured on Jetson, batch_size=4)

2000-word input → 9m 15s audio output, **RTF 0.29**, peak RAM 6.53 GB, CPU/GPU temp 66.2°C, power 17.5 W. Local Mac RTF is lower.

---

## 🔧 One-command install (macOS / Apple Silicon)

```bash
# 1. Create a conda env
conda create -n GPTSoVits python=3.10 -y
conda activate GPTSoVits

# 2. Clone this repo
git clone https://github.com/vibe570/GPT-SoVITS-Apple-Silicon-MPS.git
cd GPT-SoVITS-Apple-Silicon-MPS

# 3. Install everything (MPS mode — auto-downloads NLTK, pretrained models, etc.)
bash install.sh --device MPS --source HF      # or HF-Mirror / ModelScope

# 4. Launch the WebUI
python webui.py
```

> If `install.sh`'s NLTK download is blocked by your proxy, manually grab `cmudict` and `averaged_perceptron_tagger_eng` from GitHub and drop them into `~/nltk_data`.

### Step-by-step install (more control)

```bash
conda create -n GPTSoVits python=3.10 -y
conda activate GPTSoVits

# PyTorch (conda-forge ships MPS builds out of the box)
conda install -c conda-forge pytorch=2.10.0 torchvision=0.20.1 torchaudio=2.11.0 -y
# FFmpeg
conda install -c conda-forge ffmpeg=9 -y

# Python deps
pip install -r extra-req.txt --no-deps
pip install -r requirements.txt
```

---

## 🧪 Reproducible environment (exact versions)

Use this exact config to reproduce every benchmark number on this page.

**Hardware & system**

| Item | Value |
|------|-------|
| Chip | Apple M5 |
| RAM | 32 GB |
| Arch | `arm64` |
| macOS | 27.0 |
| Python | 3.10.21 (conda) |
| Conda | miniforge3 |
| FFmpeg | 9.0.1 |

**PyTorch + MPS**

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.10.0 | MPS built & available, `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| torchvision | 0.20.1 | |
| torchaudio | 2.11.0 | |
| cuda | unavailable | No NVIDIA GPU on Mac |
| mps_available | ✅ True | |

**Key dependencies (installed vs. requirements.txt constraints)**

| Package | Installed | Constraint | Notes |
|---------|-----------|------------|-------|
| funasr | 1.4.15 | `>=1.3.7` | OK |
| transformers | 4.57.6 | `>=4.51,<5` | OK |
| peft | 0.17.1 | `<0.18.0` | ⚠️ Close to the ceiling |
| pytorch-lightning | 2.6.6 | `>=2.4` | OK |
| gradio | 4.44.1 | `<5` | OK |
| fastapi | 0.141.1 | `>=0.115.2` | OK |
| modelscope | 1.40.0 | — | |
| librosa | 0.10.2 | `==0.10.2` | ✅ Exact match |
| numpy | 1.26.4 | `<2.0` | OK |
| torchmetrics | 1.5.0 | `<=1.5` | ✅ Pinned at upper bound |
| pydantic | 2.10.6 | `<=2.10.6` | ✅ Pinned at upper bound |
| rotary-embedding-torch | 0.9.1 | — | pip package has a hyphen, import uses underscore |

A full freeze can be exported with `pip freeze > requirements.lock` and restored with `pip install -r requirements.lock`.

**Known footguns**

- **v2Pro/v3/v3_lora + MPS + fp16 = immediate crash** — this fork already forces `is_half=False` in code; no manual config edit needed
- **don't blindly `pip install -U`** — peft / torchmetrics / pydantic all have upper-bound constraints (`<0.18.0`, `<=1.5`, `<=2.10.6`) and upgrading past them breaks things

---

## 🖥️ Usage

### Launch WebUI

```bash
# All-in-one (training + inference + UVR5 tools)
python webui.py

# Inference WebUI only
python GPT_SoVITS/inference_webui.py
```

### CLI TTS inference

```bash
# JSON POST to the local API server (start it first)
curl -X POST http://localhost:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Hello everyone, welcome to my channel.",
    "text_lang": "en",
    "ref_audio_path": "reference.wav",
    "prompt_lang": "en",
    "prompt_text": "what is being said in the reference audio",
    "text_split_method": "cut0",
    "batch_size": 4,
    "media_type": "wav",
    "streaming_mode": false
  }'
```

### CLI UVR5 vocal separation

```bash
python tools/uvr5/webui.py "mps" false 9881
```

---

## 📦 Pretrained models

`install.sh` downloads these automatically. Manual placement reference:

| Model | Path | Source |
|-------|------|--------|
| GPT-SoVITS pretrained weights | `GPT_SoVITS/pretrained_models/` | [HuggingFace](https://huggingface.co/lj1995/GPT-SoVITS) |
| G2PW Chinese text frontend | `GPT_SoVITS/text/G2PWModel/` | [HF](https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip) / [ModelScope](https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip) |
| UVR5 separation weights | `tools/uvr5/uvr5_weights/` | [HF](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights) |
| FunASR | Auto-downloaded on first use | — |
| NLTK annotation data | `~/nltk_data/` | Auto-fetched by `install.sh` |

---

## ✍️ Dataset format

Each line in a `.list` annotation file:

```
audio_path|speaker_name|language|text
```

Language codes: `zh` Chinese / `en` English / `ja` Japanese / `ko` Korean / `yue` Cantonese

---

## 🙏 Acknowledgements

This fork is developed on top of the upstream GPT-SoVITS (MIT License). All credits to the original contributors. See [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) for the full list.

Upstream builds on ar-vits / SoundStorm / vits / contentvec / hifi-gan / BigVGAN / eresnetv2 / paddlespeech / split-lang / g2pW / ultimatevocalremovergui / FunASR / Fun-ASR / SenseVoice / FFmpeg / gradio / faster-whisper and others.
