# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CosyVoice is an LLM-based multilingual zero-shot text-to-speech (TTS) system from Alibaba. Three model generations exist: v1 (CosyVoice-300M, Transformer-based), v2 (CosyVoice2-0.5B, Qwen2-based), v3 (Fun-CosyVoice3-0.5B, Qwen2 + DiT-based).

## Build & Development Commands

```sh
# Install dependencies
pip install -r requirements.txt

# Run basic inference example (uses pretrained models from modelscope/huggingface)
python example.py

# Web demo UI
python webui.py --port 50000 --model_dir pretrained_models/Fun-CosyVoice3-0.5B

# Training (LLM, Flow, or HiFiGAN)
python cosyvoice/bin/train.py \
  --model <llm|flow|hifigan> \
  --config <config.yaml> \
  --train_data <data.list> \
  --cv_data <cv.list> \
  --model_dir <output_dir> \
  --train_engine <torch_ddp|deepspeed> \
  [--qwen_pretrain_path <path>] \
  [--onnx_path <path>] \
  [--checkpoint <ckpt.pt>] \
  [--dpo --ref_model <ref.pt>]

# Export ONNX (flow decoder estimator -> TensorRT)
python cosyvoice/bin/export_onnx.py --model_dir pretrained_models/CosyVoice-300M

# Average model checkpoints
python cosyvoice/bin/average_model.py --dst_model avg.pt --src_path <ckpt_dir> --num 5

# TensorRT-LLM deployment
cd runtime/triton_trtllm && docker compose up -d
```

There is no test suite in this repository.

## Architecture

### Inference Pipeline

```
Text → Frontend (normalization + tokenization) → LLM (speech tokens) → Flow Matching (mel spectrogram) → HiFiGAN (waveform)
```

### Key Components (cosyvoice/)

- **`cli/`** — Inference entry points. `CosyVoice` / `CosyVoice2` / `CosyVoice3` classes in `cosyvoice.py` each compose a frontend + model. `AutoModel()` factory detects the model version from `cosyvoice.yaml` / `cosyvoice2.yaml` / `cosyvoice3.yaml` in the model directory. `model.py` contains `CosyVoiceModel` / `CosyVoice2Model` / `CosyVoice3Model` which orchestrate LLM → Flow → HiFiGAN with streaming support. `frontend.py` handles text normalization, speech token extraction (whisper mel → ONNX), and speaker embedding (kaldi fbank → CAM++ ONNX).

- **`llm/llm.py`** — Autoregressive speech token generation. v1 uses `TransformerLM` (conformer encoder + transformer decoder). v2 uses `Qwen2LM` (Qwen2 based, supports bi-streaming: interleaved text/speech token decoding). v3 uses `CosyVoice3LM(Qwen2LM)` adding instruct tokens, FSQ speech tokens. vLLM integration via `load_vllm()`.

- **`flow/flow.py`** — Flow matching: speech tokens → mel spectrogram. v1: `MaskedDiffWithXvec` (non-causal). v2: `CausalMaskedDiffWithXvec` (causal, streaming). v3: `CausalMaskedDiffWithDiT` (DiT decoder, no encoder needed).

- **`hifigan/`** — Vocoder: mel spectrogram → waveform. `hifigan.py` wraps generator + discriminator GAN training. `HiFTGenerator` for v1/v2, `CausalHiFTGenerator` for v3.

- **`transformer/`** — Standard building blocks (attention, convolution, positional encoding, encoder/decoder layers) borrowed from WeNet/ESPnet. Conformer and Transformer encoders used by v1 LLM.

- **`tokenizer/`** — v1 uses Whisper-style tiktoken. v2/v3 use Qwen2 `AutoTokenizer` with TTS-specific special tokens (breath, laughter, speaker tags, CMU phonemes for v3).

- **`dataset/`** — `IterableDataset` with `DistributedSampler` and a `Processor` pipeline pattern. Training data is sharded `.list` files.

- **`bin/train.py`** — Distributed training entry point. Supports torch DDP and DeepSpeed, GAN training (HiFiGAN), and DPO fine-tuning (LLM). Configs use HyperPyYAML.

- **`utils/class_utils.py`** — `get_model_type()` maps config components to model classes by isinstance checks (e.g., `Qwen2LM` + `CausalMaskedDiffWithXvec` + `HiFTGenerator` → `CosyVoice2Model`). Also registers activation, subsampling, embedding, and attention class factories.

### Inference Modes

- **SFT**: Pre-trained speaker voices (`inference_sft`)
- **Zero-shot**: Voice cloning from prompt audio + text (`inference_zero_shot`)
- **Cross-lingual**: Prompt audio in one language, synthesis text in another (`inference_cross_lingual`)
- **Instruct**: Natural language control over dialect, emotion, speed, volume (`inference_instruct` / `inference_instruct2`)
- **Voice Conversion (VC)**: v1 only (`inference_vc`)

### Configuration

Model configs (`cosyvoice.yaml`, `cosyvoice2.yaml`, `cosyvoice3.yaml`) are HyperPyYAML files stored alongside model weights. Training configs live in `examples/libritts/`. Configs define model architecture (LLM, flow, HiFiGAN), tokenizer, feature extractor, and training hyperparameters.

### Dependencies & Tooling

- PyTorch 2.3.1 + CUDA 12.1
- `transformers` (Qwen2 backbone for v2/v3 LLM)
- `deepspeed` for distributed training
- ONNX Runtime for speech tokenizer + speaker embedding inference
- vLLM for accelerated LLM decoding (v2/v3)
- TensorRT for flow decoder acceleration
- Gradio for web UI (`webui.py`)
- `third_party/Matcha-TTS` (git submodule) — HiFiGAN building blocks
- `modelscope` / `huggingface_hub` for pretrained model download

### Streaming

Set `stream=True` for chunk-based streaming TTS. v2/v3 use `token_hop_len=25` with `stream_scale_factor=2` for incremental decoding. Bi-streaming (`inference_bistream`) interleaves text input streaming with speech token generation — only for v2/v3 without vLLM.

### Deployment

- `runtime/python/grpc/` — gRPC server/client
- `runtime/python/fastapi/` — FastAPI server/client  
- `runtime/triton_trtllm/` — NVIDIA Triton + TensorRT-LLM for production serving (4x speedup on LLM)
- `cosyvoice/bin/export_onnx.py` — export flow decoder to ONNX → TensorRT engine
- `cosyvoice/bin/export_jit.py` — TorchScript JIT export for v1 LLM/flow encoder
