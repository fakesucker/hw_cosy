#!/usr/bin/env python3
"""Prepare prompt-conditioned OPSD data from dialogue.lst plus text JSONL."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import onnxruntime as ort
import torchaudio
import whisper


HW_PROJ_ROOT = Path(__file__).resolve().parents[3]


def load_prompt_scp(path: Path) -> List[Dict[str, str]]:
    prompts = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if "|" in line:
                wav, text = line.split("|", 1)
                wav = wav.strip()
                text = text.strip()
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                wav = parts[0] if Path(parts[0]).is_absolute() else parts[-1]
                text = " ".join(parts[1:-1]) if Path(parts[-1]).is_absolute() and len(parts) > 2 else " ".join(parts[1:])
                if not Path(wav).exists() and Path(parts[-1]).exists():
                    wav = parts[-1]
            if Path(wav).exists():
                prompts.append({"prompt_wav": str(Path(wav).resolve()), "prompt_text": text})
    if not prompts:
        raise ValueError(f"no valid prompt wav rows found in {path}")
    return prompts


def resolve_prompt_wav(path: str, base_dir: Path) -> str:
    wav = Path(path).expanduser()
    if not wav.is_absolute():
        wav = base_dir / wav
    return str(wav.resolve())


def ensure_speaker_tag(text: str, speaker_tag: str) -> str:
    text = text.strip()
    if not speaker_tag:
        return text
    if text.startswith("<|spk_"):
        return text
    return f"{speaker_tag}{text}"


def read_dialogue_rows(dialogue_lst: Path,
                       prompts: List[Dict[str, str]],
                       dialogue_prompt_source: str = "row",
                       fixed_prompt: Optional[Dict[str, str]] = None,
                       add_speaker_tag: bool = False,
                       speaker_tag: str = "<|spk_1|>") -> List[Dict]:
    rows = []
    base_dir = dialogue_lst.parent
    with dialogue_lst.open("r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            if fixed_prompt is not None:
                prompt = fixed_prompt
            elif dialogue_prompt_source == "row":
                prompt_wav = resolve_prompt_wav(parts[2].strip(), base_dir)
                if not Path(prompt_wav).exists():
                    raise FileNotFoundError(f"dialogue prompt wav not found: {prompt_wav}")
                prompt = {"prompt_wav": prompt_wav, "prompt_text": parts[1].strip()}
            elif dialogue_prompt_source == "cycle":
                prompt = prompts[idx % len(prompts)]
            else:
                raise ValueError(f"unsupported dialogue_prompt_source: {dialogue_prompt_source}")
            text = parts[3].strip()
            if not text:
                continue
            if add_speaker_tag:
                text = ensure_speaker_tag(text, speaker_tag)
            rows.append({
                "utt": "dialogue_{}".format(parts[0].strip() or "{:06d}".format(idx)),
                "text": text,
                "prompt_text": prompt["prompt_text"],
                "prompt_wav": prompt["prompt_wav"],
                "caption": parts[4].strip() if len(parts) > 4 else "",
                "source": "dialogue",
            })
    return rows


def read_jsonl_text_rows(jsonl_path: Path,
                         prompts: List[Dict[str, str]],
                         max_records: int = 0,
                         source_name: str = "shenhu",
                         fixed_prompt: Optional[Dict[str, str]] = None,
                         add_speaker_tag: bool = False,
                         speaker_tag: str = "<|spk_1|>") -> List[Dict]:
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if max_records > 0 and len(rows) >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            if add_speaker_tag:
                text = ensure_speaker_tag(text, speaker_tag)
            prompt = fixed_prompt if fixed_prompt is not None else prompts[len(rows) % len(prompts)]
            utt = str(item.get("utt", f"{idx:08}"))
            record = {
                "utt": f"{source_name}_{utt}",
                "text": text,
                "prompt_text": prompt["prompt_text"],
                "prompt_wav": prompt["prompt_wav"],
                "caption": str(item.get("caption", "")),
                "source": source_name,
            }
            code = item.get("code")
            if isinstance(code, list) and len(code) > 0:
                record["code"] = [int(x) for x in code]
            rows.append(record)
    return rows


def extract_prompt_token(session: ort.InferenceSession, wav_path: str) -> List[int]:
    audio, sample_rate = torchaudio.load(wav_path, backend="soundfile")
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    if audio.shape[1] / 16000 > 30:
        raise ValueError(f"prompt wav longer than 30s: {wav_path}")
    feat = whisper.log_mel_spectrogram(audio, n_mels=128)
    token = session.run(
        None,
        {
            session.get_inputs()[0].name: feat.detach().cpu().numpy(),
            session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
        },
    )[0]
    return [int(x) for x in token.reshape(-1).tolist()]


def dummy_code_len(text: str) -> int:
    return max(25, min(6000, len(text) * 4))


def finalize_records(records: Iterable[Dict], prompt_tokens: Dict[str, List[int]]) -> List[Dict]:
    finalized = []
    for item in records:
        record = dict(item)
        if "code" not in record:
            record["code"] = [0] * dummy_code_len(record["text"])
            record["dummy_speech_token"] = True
        else:
            record["dummy_speech_token"] = False
        record["prompt_speech_token"] = prompt_tokens[record["prompt_wav"]]
        finalized.append(record)
    return finalized


def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_list(path: Path, jsonl_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        fout.write(str(jsonl_path.resolve()) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogue-lst", type=Path,
                        default=HW_PROJ_ROOT / "testset_midterm/cmos/dialogue.lst")
    parser.add_argument("--extra-jsonl", type=Path,
                        default=HW_PROJ_ROOT / "CosyVoice/data_list/shenhu/shenhu_filtered_wo_outliers98.jsonl")
    parser.add_argument("--prompt-scp", type=Path,
                        default=Path("/home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/prompt_wav/utt_text.scp"))
    parser.add_argument("--onnx-path", type=Path,
                        default=Path("/home/work_nfs23/hkxie/code/CosyVoice/pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "data/opsd_dialogue_shenhu")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--seed", type=int, default=1986)
    parser.add_argument("--cv-size", type=int, default=200)
    parser.add_argument("--max-extra-records", type=int, default=0)
    parser.add_argument("--dialogue-prompt-source", choices=["row", "cycle"], default="row",
                        help="row uses prompt_text/prompt_wav from dialogue.lst; cycle uses --prompt-scp round-robin")
    parser.add_argument("--fixed-prompt-wav", type=Path, default=None,
                        help="If set with --fixed-prompt-text, use this prompt wav for all records")
    parser.add_argument("--fixed-prompt-text", default="",
                        help="If set with --fixed-prompt-wav, use this prompt text for all records")
    parser.add_argument("--add-speaker-tag", action="store_true",
                        help="Write the speaker tag into JSONL text. Training configs also add it if missing.")
    parser.add_argument("--speaker-tag", default="<|spk_1|>")
    args = parser.parse_args()

    prompts = load_prompt_scp(args.prompt_scp)
    fixed_prompt = None
    if args.fixed_prompt_wav is not None or args.fixed_prompt_text:
        if args.fixed_prompt_wav is None or not args.fixed_prompt_text:
            raise ValueError("--fixed-prompt-wav and --fixed-prompt-text must be set together")
        fixed_prompt_wav = str(args.fixed_prompt_wav.expanduser().resolve())
        if not Path(fixed_prompt_wav).exists():
            raise FileNotFoundError(f"--fixed-prompt-wav not found: {fixed_prompt_wav}")
        fixed_prompt = {"prompt_wav": fixed_prompt_wav, "prompt_text": args.fixed_prompt_text.strip()}
    dialogue_rows = read_dialogue_rows(
        args.dialogue_lst,
        prompts,
        dialogue_prompt_source=args.dialogue_prompt_source,
        fixed_prompt=fixed_prompt,
        add_speaker_tag=args.add_speaker_tag,
        speaker_tag=args.speaker_tag,
    )
    extra_rows = read_jsonl_text_rows(
        args.extra_jsonl,
        prompts,
        args.max_extra_records,
        fixed_prompt=fixed_prompt,
        add_speaker_tag=args.add_speaker_tag,
        speaker_tag=args.speaker_tag,
    )
    all_rows = dialogue_rows + extra_rows
    if not all_rows:
        raise ValueError("no records prepared")

    session = ort.InferenceSession(str(args.onnx_path), providers=[args.provider])
    prompt_wavs = sorted({record["prompt_wav"] for record in all_rows})
    prompt_tokens = {
        prompt_wav: extract_prompt_token(session, prompt_wav)
        for prompt_wav in prompt_wavs
    }
    records = finalize_records(all_rows, prompt_tokens)
    rng = random.Random(args.seed)
    rng.shuffle(records)

    cv_size = min(max(args.cv_size, 0), max(len(records) - 1, 0))
    cv_records = records[:cv_size]
    train_records = records[cv_size:]

    train_jsonl = args.output_dir / "train.jsonl"
    cv_jsonl = args.output_dir / "cv.jsonl"
    train_list = args.output_dir / "train.data.list"
    cv_list = args.output_dir / "cv.data.list"
    write_jsonl(train_jsonl, train_records)
    write_jsonl(cv_jsonl, cv_records)
    write_list(train_list, train_jsonl)
    write_list(cv_list, cv_jsonl)

    print(f"dialogue_records={len(dialogue_rows)}")
    print(f"extra_records={len(extra_rows)}")
    print(f"records={len(records)}")
    print(f"train_records={len(train_records)}")
    print(f"cv_records={len(cv_records)}")
    print(f"unique_prompt_wavs={len(prompt_tokens)}")
    print(f"dialogue_prompt_source={args.dialogue_prompt_source}")
    print(f"fixed_prompt_wav={fixed_prompt['prompt_wav'] if fixed_prompt is not None else ''}")
    print(f"add_speaker_tag={args.add_speaker_tag} speaker_tag={args.speaker_tag}")
    print(f"train_data={train_list.resolve()}")
    print(f"cv_data={cv_list.resolve()}")


if __name__ == "__main__":
    main()
