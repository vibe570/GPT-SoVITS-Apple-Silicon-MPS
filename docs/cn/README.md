<div align="center">

<h1>GPT-SoVITS · Apple Silicon MPS 优化版</h1>
Apple Silicon (M1/M2/M3/M4/M5) 专属优化的 GPT-SoVITS fork — 端到端 MPS fp32 推理、训练、人声分离全加速，无需 CUDA。<br><br>

[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge&logo=python)](https://www.python.org)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge&logo=opensourceinitiative)](https://github.com/vibe570/GPT-SoVITS-Apple-Silicon-MPS/blob/main/LICENSE)

[**English**](../../README.md) | **中文简体**

</div>

---

## 🍎 这个 fork 做了什么？

基于官方 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 主线，对 Apple Silicon 做了**端到端 MPS 适配**。官方主线在 macOS 上基本只能用 CPU；本 fork 把推理、训练、数据预处理、人声分离（UVR5）、ASR（FunASR）全部打通到 MPS。clone 后一条 `install.sh --device MPS` 即可运行，无需改代码、无需 CUDA。

### 与官方上游的差异

| 能力 | 官方上游 | 本 fork |
|------|----------|---------|
| macOS TTS 推理 | 仅 CPU | **自动检测 MPS，fp32 推理**（v2Pro/v3 权重 fp16 会崩溃，故强制 fp32） |
| macOS 训练 (s1/s2) | 未专门适配 | **s1/s2/s2_v3_lora 全链路 MPS fp32 训练** |
| 数据预处理 (hubert/sv/semantic/get-text) | 未专门适配 | **MPS 自动回退，无需改代码** |
| 人声分离 UVR5 | 未专门适配 | **MPS fp32 权重 + fp16 autocast 计算**，避免矩阵乘法崩溃同时保证精度 |
| FunASR ASR | 未专门适配 | **MPS 设备自动选择** |
| `PYTORCH_ENABLE_MPS_FALLBACK` | 需手动 export | **自动 `setdefault`**，未实现算子兜底回退 CPU |

核心改动文件：`GPT_SoVITS/TTS_infer_pack/TTS.py`、`config.py`、`GPT_SoVITS/inference_webui.py`、`s1_train.py` / `s2_train.py` / `s2_train_v3_lora.py`、`prepare_datasets/*.py`、`tools/asr/funasr_asr.py`。

---

## ⚡ 实测效果（Apple M5 / 32GB RAM）

### 1. TTS 推理（v2ProPlus 权重）

| 配置 | 27 字符推理耗时 | 说明 |
|------|----------------|------|
| 官方原始 MPS fp32 | 7.08 s | — |
| **本 fork 优化后 MPS fp32** | **6.79 s** | 约 4% 加速 |
| CPU 回退 | ~60+ s | 参考 |

MPS 上 `v2Pro` / `v3` / `v3_lora` 权重**fp16 半精度会直接崩溃**，本 fork 自动 `is_half=False`，用户无需手动改配置。

### 2. UVR5 人声分离（bs_roformer）

| 配置 | 单 chunk 耗时 | SNR（端到端） |
|------|--------------|---------------|
| 官方 CPU | 31 ~ 61 s/chunk | 基准 |
| **本 fork MPS** | **9.6 s/chunk** | **68.7 dB**（听感无损） |

约 **3–6× 加速**，通过"权重 fp32 + 计算 fp16 autocast"策略，矩阵乘法不再触发 MPS 断言崩溃，精度损失可忽略。

### 3. 长文本 TTS（同项目 Jetson 侧实测，batch_size=4）

2000 字文本合成 → 9 分 15 秒音频，**RTF 0.29**，峰值 RAM 6.53 GB，CPU/GPU 温度 66.2°C，功耗 17.5W。本地 Mac 推理 RTF 更低。

---

## 🔧 一键安装（macOS / Apple Silicon）

```bash
# 1. 创建 conda 环境
conda create -n GPTSoVits python=3.10 -y
conda activate GPTSoVits

# 2. clone 本仓库
git clone https://github.com/vibe570/GPT-SoVITS-Apple-Silicon-MPS.git
cd GPT-SoVITS-Apple-Silicon-MPS

# 3. 一键安装（MPS 模式，自动拉取 NLTK、预训练模型等）
bash install.sh --device MPS --source HF      # 或 HF-Mirror / ModelScope

# 4. 启动 WebUI
python webui.py
```

> 若 `install.sh` 下载 NLTK 数据被网络代理拦截，可手动从 GitHub 拉取 `cmudict` 与 `averaged_perceptron_tagger_eng` 到 `~/nltk_data`。

### 手动逐行装（更可控）

```bash
conda create -n GPTSoVits python=3.10 -y
conda activate GPTSoVits

# PyTorch (用 conda-forge 装，自动带 MPS)
conda install -c conda-forge pytorch=2.10.0 torchvision=0.20.1 torchaudio=2.11.0 -y
# FFmpeg
conda install -c conda-forge ffmpeg=9 -y

# Python 依赖
pip install -r extra-req.txt --no-deps
pip install -r requirements.txt
```

---

## 🧪 实测环境复现（精确版本）

按此配置可 1:1 复现 README 中的所有性能数据。

**硬件与系统**

| 项 | 值 |
|----|----|
| 芯片 | Apple M5 |
| 内存 | 32 GB |
| 架构 | `arm64` |
| macOS | 27.0 |
| Python | 3.10.21（conda） |
| Conda | miniforge3 |
| FFmpeg | 9.0.1 |

**PyTorch + MPS**

| 包 | 版本 | 备注 |
|----|------|------|
| torch | 2.10.0 | MPS 已编译并可用，`PYTORCH_ENABLE_MPS_FALLBACK=1` |
| torchvision | 0.20.1 | |
| torchaudio | 2.11.0 | |
| cuda | 不可用 | Mac 无 NVIDIA GPU |
| mps_available | ✅ True | |

**关键依赖（实测版本 vs requirements 约束）**

| 包 | 实测版本 | requirements 约束 | 说明 |
|----|----------|-------------------|------|
| funasr | 1.4.15 | `>=1.3.7` | 满足 |
| transformers | 4.57.6 | `>=4.51,<5` | 满足 |
| peft | 0.17.1 | `<0.18.0` | ⚠️ 接近上限 |
| pytorch-lightning | 2.6.6 | `>=2.4` | 满足 |
| gradio | 4.44.1 | `<5` | 满足 |
| fastapi | 0.141.1 | `>=0.115.2` | 满足 |
| modelscope | 1.40.0 | — | |
| librosa | 0.10.2 | `==0.10.2` | ✅ 精确对齐 |
| numpy | 1.26.4 | `<2.0` | 满足 |
| torchmetrics | 1.5.0 | `<=1.5` | ✅ 卡在上限 |
| pydantic | 2.10.6 | `<=2.10.6` | ✅ 卡在上限 |
| rotary-embedding-torch | 0.9.1 | — | pip 包名带横线，import 名带下划线 |

更多依赖完整列表可从 `pip freeze > requirements.lock` 导出后装回。

**已知版本坑**

- **v2Pro/v3/v3_lora + MPS + fp16 = 直接崩溃**：本 fork 已在代码层强制 `is_half=False`，用户无需手动改配置
- **peft / torchmetrics / pydantic 别盲目升级**：requirements 里有上限约束（`<0.18.0`、`<=1.5`、`<=2.10.6`），`pip install -U` 容易炸

---

## 🖥️ 使用

### 启动 WebUI

```bash
# 一键启动（集成训练 + 推理 + 人声分离工具）
python webui.py

# 只启动推理 WebUI
python GPT_SoVITS/inference_webui.py
```

### 命令行 TTS 推理

```bash
# JSON POST 方式调用本地 API（启动服务端后）
curl -X POST http://localhost:9880/tts \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "你好呀，这是 MPS 加速测试。",
    "text_lang": "zh",
    "ref_audio_path": "参考音频.wav",
    "prompt_lang": "zh",
    "prompt_text": "参考音频里说的话",
    "text_split_method": "cut0",
    "batch_size": 4,
    "media_type": "wav",
    "streaming_mode": false
  }'
```

### 命令行 UVR5 人声分离

```bash
python tools/uvr5/webui.py "mps" false 9881
```

---

## 📦 预训练模型

`install.sh` 成功后会自动下载，以下为手动放置参考：

| 模型 | 放置路径 | 来源 |
|------|----------|------|
| GPT-SoVITS 预训练权重 | `GPT_SoVITS/pretrained_models/` | [HuggingFace](https://huggingface.co/lj1995/GPT-SoVITS) |
| G2PW 中文文本前端 | `GPT_SoVITS/text/G2PWModel/` | [HF](https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip) / [ModelScope](https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip) |
| UVR5 人声分离权重 | `tools/uvr5/uvr5_weights/` | [HF](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights) |
| FunASR | 首次使用自动下载 | — |
| NLTK 标注数据 | `~/nltk_data/` | `install.sh` 自动拉取 |

---

## ✍️ 数据集格式

标注 `.list` 文件每行格式：

```
音频路径|说话人|语言|文本
```

语言代码：`zh` 中文 / `en` 英文 / `ja` 日文 / `ko` 韩文 / `yue` 粤语

---

## 🙏 致谢

本 fork 基于官方 GPT-SoVITS（MIT License）二次开发，感谢原项目所有贡献者。完整 credits 见上游仓库 [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)。

上游引用的核心项目：ar-vits / SoundStorm / vits / contentvec / hifi-gan / BigVGAN / eresnetv2 / paddlespeech / split-lang / g2pW / ultimatevocalremovergui / FunASR / Fun-ASR / SenseVoice / FFmpeg / gradio / faster-whisper 等。
