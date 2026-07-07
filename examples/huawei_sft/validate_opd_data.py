#!/usr/bin/env python3
# Copyright (c) 2026
#
# Lightweight data preflight for CosyVoice2 LLM OPD distillation.

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


JSONL_SUFFIXES = {".jsonl", ".json", ".txt"}


def read_list_file(path: Path) -> List[Path]:
    entries = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            entries.append(Path(line))
    return entries


def iter_record_files(entry: Path, args) -> Iterable[Path]:
    if entry.is_file():
        yield entry
        return
    if not entry.is_dir():
        return
    yielded = 0
    for child in entry.rglob("*"):
        if not child.is_file():
            continue
        if child.suffix.lower() not in JSONL_SUFFIXES:
            continue
        yield child
        yielded += 1
        if args.max_files_per_entry > 0 and yielded >= args.max_files_per_entry:
            break


def parse_speech_token(value: Any) -> Tuple[List[int], str]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            return [], "speech_token literal parse failed: {}".format(exc)
    if not isinstance(value, list):
        return [], "speech_token/code must be a list, got {}".format(type(value).__name__)
    tokens = []
    for index, token in enumerate(value):
        if isinstance(token, bool) or not isinstance(token, int):
            return [], "speech_token[{}] must be int, got {}".format(index, type(token).__name__)
        tokens.append(token)
    return tokens, ""


def empty_stats() -> Dict[str, Any]:
    return {
        "list_entries": 0,
        "files_seen": 0,
        "missing_files": 0,
        "records_seen": 0,
        "records_checked": 0,
        "valid_records": 0,
        "hard_errors": 0,
        "warnings": 0,
        "token_lens": [],
        "text_lens": [],
        "examples": [],
    }


def add_example(stats: Dict[str, Any], examples_limit: int, message: str) -> None:
    if len(stats["examples"]) < examples_limit:
        stats["examples"].append(message)


def validate_record(record: Dict[str, Any],
                    jsonl_path: Path,
                    line_no: int,
                    split: str,
                    args,
                    stats: Dict[str, Any]) -> None:
    utt = record.get("key", record.get("utt", ""))
    text = record.get("txt", record.get("text", ""))
    speech_value = record.get("code", record.get("speech_token", []))
    prompt_speech_value = record.get("prompt_code", record.get("prompt_speech_token", []))

    prefix = "{}:{}:{}: ".format(split, jsonl_path, line_no)
    errors = []
    warnings = []

    if not utt:
        errors.append("missing key/utt")
    if not isinstance(text, str) or len(text.strip()) == 0:
        errors.append("missing or empty txt/text")
    if args.require_prompt_fields:
        prompt_text = record.get("prompt_text", "")
        prompt_wav = record.get("prompt_wav", "")
        if not isinstance(prompt_text, str) or len(prompt_text.strip()) == 0:
            errors.append("missing or empty prompt_text")
        if not isinstance(prompt_wav, str) or len(prompt_wav.strip()) == 0:
            errors.append("missing or empty prompt_wav")
        elif not Path(prompt_wav).is_file():
            errors.append("prompt_wav not found: {}".format(prompt_wav))

    speech_token, token_error = parse_speech_token(speech_value)
    if token_error:
        errors.append(token_error)
    elif len(speech_token) == 0:
        errors.append("empty code/speech_token")
    else:
        bad_range = [token for token in speech_token if token < 0 or token >= args.speech_token_size]
        if bad_range:
            errors.append(
                "speech_token out of range [0, {}), first_bad={}".format(args.speech_token_size, bad_range[0]))

    if args.require_prompt_fields:
        prompt_speech_token, prompt_token_error = parse_speech_token(prompt_speech_value)
        if prompt_token_error:
            errors.append("prompt_{}".format(prompt_token_error))
        elif len(prompt_speech_token) == 0:
            errors.append("empty prompt_code/prompt_speech_token")
        else:
            bad_prompt_range = [token for token in prompt_speech_token if token < 0 or token >= args.speech_token_size]
            if bad_prompt_range:
                errors.append("prompt_speech_token out of range [0, {}), first_bad={}".format(
                    args.speech_token_size, bad_prompt_range[0]))

    if "duration" in record:
        try:
            duration = float(record["duration"])
            if not math.isfinite(duration) or duration <= 0:
                warnings.append("duration is not positive finite: {}".format(record["duration"]))
        except (TypeError, ValueError):
            warnings.append("duration is not numeric: {}".format(record["duration"]))

    if not errors and speech_token:
        stats["token_lens"].append(len(speech_token))
        stats["text_lens"].append(len(text))
        if len(speech_token) < args.warn_min_speech_tokens:
            warnings.append("short speech_token length {}".format(len(speech_token)))
        if len(speech_token) > args.warn_max_speech_tokens:
            warnings.append("long speech_token length {}".format(len(speech_token)))

    if errors:
        stats["hard_errors"] += 1
        add_example(stats, args.examples, prefix + "; ".join(errors))
        return

    if warnings:
        stats["warnings"] += len(warnings)
        add_example(stats, args.examples, prefix + "; ".join(warnings))

    stats["valid_records"] += 1


def validate_split(split: str, data_list: Path, args) -> Dict[str, Any]:
    stats = empty_stats()
    if not data_list.is_file():
        stats["hard_errors"] += 1
        add_example(stats, args.examples, "{} list file not found: {}".format(split, data_list))
        return stats

    jsonl_paths = read_list_file(data_list)
    stats["list_entries"] = len(jsonl_paths)
    if len(jsonl_paths) == 0:
        stats["hard_errors"] += 1
        add_example(stats, args.examples, "{} list file is empty: {}".format(split, data_list))
        return stats

    files_opened = 0
    for entry in jsonl_paths:
        if args.max_records_per_split > 0 and stats["records_checked"] >= args.max_records_per_split:
            break
        found_entry_file = False
        for jsonl_path in iter_record_files(entry, args):
            found_entry_file = True
            if args.max_files_per_split > 0 and files_opened >= args.max_files_per_split:
                break
            if args.max_records_per_split > 0 and stats["records_checked"] >= args.max_records_per_split:
                break
            stats["files_seen"] += 1
            files_opened += 1
            with jsonl_path.open("r", encoding="utf-8") as fin:
                for line_no, line in enumerate(fin, start=1):
                    stats["records_seen"] += 1
                    line = line.strip()
                    if not line:
                        continue
                    if args.max_records_per_split > 0 and stats["records_checked"] >= args.max_records_per_split:
                        break
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        stats["hard_errors"] += 1
                        add_example(
                            stats,
                            args.examples,
                            "{}:{}:{}: JSON parse failed: {}".format(split, jsonl_path, line_no, exc),
                        )
                        stats["records_checked"] += 1
                        continue
                    if not isinstance(record, dict):
                        stats["hard_errors"] += 1
                        add_example(stats, args.examples, "{}:{}:{}: JSON record must be object".format(
                            split, jsonl_path, line_no))
                        stats["records_checked"] += 1
                        continue
                    validate_record(record, jsonl_path, line_no, split, args, stats)
                    stats["records_checked"] += 1

        if not found_entry_file:
            stats["missing_files"] += 1
            stats["hard_errors"] += 1
            add_example(stats, args.examples, "{} data entry not found or has no JSONL files: {}".format(split, entry))
            continue

        if args.max_files_per_split > 0 and files_opened >= args.max_files_per_split:
            break

    if stats["records_checked"] == 0:
        stats["hard_errors"] += 1
        add_example(stats, args.examples, "{} has no records checked from {}".format(split, data_list))
    if stats["valid_records"] == 0:
        stats["hard_errors"] += 1
        add_example(stats, args.examples, "{} has no valid OPD records".format(split))
    return stats


def mean(values: List[int]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def format_len_summary(values: List[int]) -> str:
    if not values:
        return "count=0"
    return "count={} min={} max={} mean={:.2f}".format(len(values), min(values), max(values), mean(values))


def print_split(split: str, stats: Dict[str, Any]) -> None:
    print("[{}] list_entries={} files_seen={} missing_files={} records_seen={} records_checked={} valid_records={} hard_errors={} warnings={}".format(
        split,
        stats["list_entries"],
        stats["files_seen"],
        stats["missing_files"],
        stats["records_seen"],
        stats["records_checked"],
        stats["valid_records"],
        stats["hard_errors"],
        stats["warnings"],
    ))
    print("[{}] speech_token_len {}".format(split, format_len_summary(stats["token_lens"])))
    print("[{}] text_char_len {}".format(split, format_len_summary(stats["text_lens"])))
    for example in stats["examples"]:
        print("[{}] example: {}".format(split, example))


def parse_args():
    parser = argparse.ArgumentParser(description="Validate CosyVoice2 OPD distillation data lists")
    parser.add_argument("--train-data", required=True, help="TRAIN_DATA list file")
    parser.add_argument("--cv-data", required=True, help="CV_DATA list file")
    parser.add_argument("--speech-token-size", type=int, default=6561)
    parser.add_argument("--max-records-per-split", type=int, default=2000,
                        help="Maximum JSONL records to check per split; <=0 checks all records")
    parser.add_argument("--max-files-per-split", type=int, default=0,
                        help="Maximum JSONL files to open per split; <=0 means no file limit")
    parser.add_argument("--max-files-per-entry", type=int, default=100,
                        help="When a list entry is a directory, maximum JSONL files to use from that directory")
    parser.add_argument("--warn-min-speech-tokens", type=int, default=25)
    parser.add_argument("--warn-max-speech-tokens", type=int, default=1500)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--require-prompt-fields", action="store_true",
                        help="Require prompt_text/prompt_wav/prompt_speech_token for OPSD data")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.speech_token_size <= 0:
        print("speech-token-size must be > 0", file=sys.stderr)
        return 2
    splits = {
        "train": Path(args.train_data),
        "cv": Path(args.cv_data),
    }
    exit_code = 0
    for split, data_list in splits.items():
        stats = validate_split(split, data_list, args)
        print_split(split, stats)
        if stats["hard_errors"] > 0:
            exit_code = 1
    if exit_code == 0:
        print("opd_data_validation_ok")
    else:
        print("opd_data_validation_failed", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
