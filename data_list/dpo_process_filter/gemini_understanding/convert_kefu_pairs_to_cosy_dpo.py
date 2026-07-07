#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert kefu_dpo_pairs.jsonl to CosyVoice DPO dataset format."
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Input kefu_dpo_pairs.jsonl path.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for converted dataset.",
    )
    parser.add_argument(
        "--cv_ratio",
        type=float,
        default=0.02,
        help="Validation split ratio. Default: 0.02",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="cosyvoice_dpo_v1",
        help="Seed string used for stable hash split.",
    )
    parser.add_argument(
        "--min_token_len",
        type=int,
        default=1,
        help="Drop pair if chosen/rejected token length < min_token_len.",
    )
    return parser.parse_args()


def _normalize_tokens(tokens: Any) -> Optional[List[int]]:
    if tokens is None:
        return None
    if isinstance(tokens, str):
        return None
    if not isinstance(tokens, list):
        return None
    out: List[int] = []
    for x in tokens:
        try:
            out.append(int(x))
        except (ValueError, TypeError):
            return None
    return out


def _pick_text(obj: Dict[str, Any]) -> str:
    prompt = obj.get("prompt")
    chosen = obj.get("chosen", {})
    text = chosen.get("text")
    if isinstance(text, str) and text.strip():
        return text
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    return ""


def _to_train_record(obj: Dict[str, Any], min_token_len: int) -> Tuple[Optional[Dict[str, Any]], str]:
    chosen = obj.get("chosen", {})
    rejected = obj.get("rejected", {})
    chosen_tokens = _normalize_tokens(chosen.get("token"))
    rejected_tokens = _normalize_tokens(rejected.get("token"))
    if chosen_tokens is None or rejected_tokens is None:
        return None, "bad_token_type"
    if len(chosen_tokens) < min_token_len or len(rejected_tokens) < min_token_len:
        return None, "token_too_short"

    utt = str(obj.get("utt", "")).strip() or str(obj.get("group_id", "")).strip()
    if not utt:
        return None, "missing_utt"

    text = _pick_text(obj)
    if not text:
        return None, "missing_text"

    rec: Dict[str, Any] = {
        "key": utt,
        "txt": text,
        "code": chosen_tokens,
        "reject_code": rejected_tokens,
        # Optional fields below are not required by training loader, but useful for audit/debug.
        "chosen_utt": chosen.get("utt", ""),
        "reject_utt": rejected.get("utt", ""),
        "chosen_wav_path": chosen.get("wav_path", ""),
        "reject_wav_path": rejected.get("wav_path", ""),
        "group_id": obj.get("group_id", ""),
    }
    return rec, ""


def _in_cv_split(key: str, seed: str, cv_ratio: float) -> bool:
    digest = hashlib.md5(f"{seed}::{key}".encode("utf-8")).hexdigest()
    val = int(digest[:8], 16) / float(0xFFFFFFFF)
    return val < cv_ratio


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl = out_dir / "dpo_train.jsonl"
    cv_jsonl = out_dir / "dpo_cv.jsonl"
    train_list = out_dir / "dpo_train.list"
    cv_list = out_dir / "dpo_cv.list"
    bad_jsonl = out_dir / "dpo_bad_samples.jsonl"
    summary_json = out_dir / "dpo_convert_summary.json"

    stats = {
        "total_lines": 0,
        "valid_pairs": 0,
        "train_pairs": 0,
        "cv_pairs": 0,
        "bad_pairs": 0,
        "bad_reasons": {},
    }

    with (
        open(in_path, "r", encoding="utf-8") as fin,
        open(train_jsonl, "w", encoding="utf-8") as f_train,
        open(cv_jsonl, "w", encoding="utf-8") as f_cv,
        open(bad_jsonl, "w", encoding="utf-8") as f_bad,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["total_lines"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_pairs"] += 1
                stats["bad_reasons"]["json_decode_error"] = stats["bad_reasons"].get("json_decode_error", 0) + 1
                f_bad.write(json.dumps({"reason": "json_decode_error", "raw": line}, ensure_ascii=False) + "\n")
                continue

            rec, reason = _to_train_record(obj, args.min_token_len)
            if rec is None:
                stats["bad_pairs"] += 1
                stats["bad_reasons"][reason] = stats["bad_reasons"].get(reason, 0) + 1
                f_bad.write(json.dumps({"reason": reason, "raw": obj}, ensure_ascii=False) + "\n")
                continue

            stats["valid_pairs"] += 1
            if _in_cv_split(rec["key"], args.seed, args.cv_ratio):
                f_cv.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["cv_pairs"] += 1
            else:
                f_train.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["train_pairs"] += 1

    with open(train_list, "w", encoding="utf-8") as f:
        f.write(str(train_jsonl) + "\n")
    with open(cv_list, "w", encoding="utf-8") as f:
        f.write(str(cv_jsonl) + "\n")

    summary = {
        **stats,
        "input_jsonl": str(in_path),
        "out_dir": str(out_dir),
        "train_jsonl": str(train_jsonl),
        "cv_jsonl": str(cv_jsonl),
        "train_list": str(train_list),
        "cv_list": str(cv_list),
        "bad_jsonl": str(bad_jsonl),
        "cv_ratio": args.cv_ratio,
        "seed": args.seed,
        "min_token_len": args.min_token_len,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
