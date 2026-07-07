#!/usr/bin/env python3
"""Prepare prompt-conditioned OPSD data from dialogue.lst metadata."""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime
import torch
import torchaudio
import whisper


SCRIPT_DIR = Path(__file__).resolve().parent
HW_PROJ_ROOT = SCRIPT_DIR.parents[2]


def normalize_key(path: str) -> str:
    path_obj = Path(path).expanduser()
    try:
        return str(path_obj.resolve())
    except OSError:
        return str(path_obj)


def resolve_prompt_wav(path: str, base_dir: Path) -> str:
    path_obj = Path(path).expanduser()
    if not path_obj.is_absolute():
        path_obj = base_dir / path_obj
    return normalize_key(str(path_obj))


def ensure_speaker_tag(text: str, speaker_tag: str) -> str:
    text = text.strip()
    if not speaker_tag:
        return text
    if text.startswith("<|spk_"):
        return text
    return "{}{}".format(speaker_tag, text)


def read_prompt_scp(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    by_path, by_name = {}, {}
    with path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|", 1)
            if len(parts) != 2:
                raise ValueError("{}:{} expected wav|text".format(path, line_no))
            wav, text = parts[0].strip(), parts[1].strip()
            by_path[normalize_key(wav)] = text
            by_name[Path(wav).name] = text
    return by_path, by_name


def read_dialogue(path: Path,
                  prompt_by_path: Dict[str, str],
                  prompt_by_name: Dict[str, str],
                  prompt_text_override: str = "",
                  prompt_wav_override: str = "",
                  add_speaker_tag: bool = False,
                  speaker_tag: str = "<|spk_1|>") -> List[dict]:
    records = []
    base_dir = path.parent
    with path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                raise ValueError("{}:{} expected at least 4 pipe-separated columns".format(path, line_no))
            utt, row_prompt_text, prompt_wav, target_text = [item.strip() for item in parts[:4]]
            prompt_key = normalize_key(prompt_wav_override) if prompt_wav_override else resolve_prompt_wav(prompt_wav, base_dir)
            prompt_text = prompt_text_override.strip() if prompt_text_override else prompt_by_path.get(
                prompt_key, prompt_by_name.get(Path(prompt_key).name, row_prompt_text))
            if add_speaker_tag:
                target_text = ensure_speaker_tag(target_text, speaker_tag)
            records.append({
                "utt": utt,
                "text": target_text,
                "prompt_text": prompt_text,
                "prompt_wav": prompt_key,
                "caption": parts[4].strip() if len(parts) > 4 else "",
            })
    return records


def resolve_onnx_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_dir():
        path = path / "speech_tokenizer_v2.onnx"
    if not path.is_file():
        raise FileNotFoundError("speech tokenizer ONNX not found: {}".format(path))
    return path


def build_onnx_session(onnx_path: Path, provider: str):
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = [provider]
    if provider == "CUDAExecutionProvider":
        providers.append("CPUExecutionProvider")
    return onnxruntime.InferenceSession(str(onnx_path), sess_options=option, providers=providers)


def extract_prompt_token(session, wav_path: str) -> List[int]:
    audio, sample_rate = torchaudio.load(wav_path, backend="soundfile")
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    if audio.shape[1] / 16000 > 30:
        raise ValueError("prompt wav longer than 30s: {}".format(wav_path))
    feat = whisper.log_mel_spectrogram(audio, n_mels=128)
    token = session.run(
        None,
        {
            session.get_inputs()[0].name: feat.detach().cpu().numpy(),
            session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
        },
    )[0].flatten().tolist()
    return [int(item) for item in token]


def dummy_speech_token(text: str, tokens_per_char: float, min_tokens: int, max_tokens: int) -> List[int]:
    token_count = int(math.ceil(max(len(text), 1) * tokens_per_char))
    token_count = max(min_tokens, min(max_tokens, token_count))
    return [0] * token_count


def write_jsonl(path: Path, records: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_data_list(path: Path, jsonl_path: Path) -> None:
    with path.open("w", encoding="utf-8") as fout:
        fout.write(str(jsonl_path.resolve()) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare prompt-conditioned OPSD dialogue data")
    parser.add_argument("--dialogue-lst", default=str(HW_PROJ_ROOT / "testset_midterm/cmos/dialogue.lst"))
    parser.add_argument("--prompt-scp",
                        default="/home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/prompt_wav/utt_text.scp")
    parser.add_argument("--onnx-path",
                        default="/home/work_nfs23/hkxie/code/CosyVoice/pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx")
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "data/opsd_dialogue"))
    parser.add_argument("--provider", default="CUDAExecutionProvider",
                        choices=["CUDAExecutionProvider", "CPUExecutionProvider"])
    parser.add_argument("--cv-size", type=int, default=16)
    parser.add_argument("--dummy-tokens-per-char", type=float, default=4.0)
    parser.add_argument("--min-dummy-tokens", type=int, default=25)
    parser.add_argument("--max-dummy-tokens", type=int, default=1200)
    parser.add_argument("--prompt-text-override", default="")
    parser.add_argument("--prompt-wav-override", default="")
    parser.add_argument("--add-speaker-tag", action="store_true",
                        help="Write the speaker tag into JSONL text. Training configs also add it if missing.")
    parser.add_argument("--speaker-tag", default="<|spk_1|>")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_dummy_tokens <= 0 or args.max_dummy_tokens < args.min_dummy_tokens:
        raise ValueError("--min-dummy-tokens must be > 0 and <= --max-dummy-tokens")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_by_path, prompt_by_name = read_prompt_scp(Path(args.prompt_scp))
    prompt_wav_override = normalize_key(args.prompt_wav_override) if args.prompt_wav_override else ""
    if prompt_wav_override and not Path(prompt_wav_override).exists():
        raise FileNotFoundError("--prompt-wav-override not found: {}".format(prompt_wav_override))
    records = read_dialogue(
        Path(args.dialogue_lst),
        prompt_by_path,
        prompt_by_name,
        prompt_text_override=args.prompt_text_override,
        prompt_wav_override=prompt_wav_override,
        add_speaker_tag=args.add_speaker_tag,
        speaker_tag=args.speaker_tag,
    )
    onnx_path = resolve_onnx_path(Path(args.onnx_path))
    session = build_onnx_session(onnx_path, args.provider)

    prompt_cache = {}
    for prompt_wav in sorted({record["prompt_wav"] for record in records}):
        prompt_cache[prompt_wav] = extract_prompt_token(session, prompt_wav)

    prepared = []
    for record in records:
        item = dict(record)
        item["code"] = dummy_speech_token(
            record["text"], args.dummy_tokens_per_char, args.min_dummy_tokens, args.max_dummy_tokens)
        item["dummy_speech_token"] = True
        item["prompt_speech_token"] = prompt_cache[record["prompt_wav"]]
        prepared.append(item)

    cv_size = min(max(args.cv_size, 0), max(len(prepared) - 1, 0))
    train_records = prepared[:-cv_size] if cv_size > 0 else prepared
    cv_records = prepared[-cv_size:] if cv_size > 0 else prepared[:1]

    train_jsonl = output_dir / "train.jsonl"
    cv_jsonl = output_dir / "cv.jsonl"
    train_list = output_dir / "train.data.list"
    cv_list = output_dir / "cv.data.list"
    write_jsonl(train_jsonl, train_records)
    write_jsonl(cv_jsonl, cv_records)
    write_data_list(train_list, train_jsonl)
    write_data_list(cv_list, cv_jsonl)

    print("records={}".format(len(prepared)))
    print("unique_prompt_wavs={}".format(len(prompt_cache)))
    print("prompt_text_override={}".format(args.prompt_text_override))
    print("prompt_wav_override={}".format(prompt_wav_override))
    print("add_speaker_tag={} speaker_tag={}".format(args.add_speaker_tag, args.speaker_tag))
    print("train_jsonl={}".format(train_jsonl.resolve()))
    print("cv_jsonl={}".format(cv_jsonl.resolve()))
    print("train_data={}".format(train_list.resolve()))
    print("cv_data={}".format(cv_list.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
