#!/usr/bin/env python3
"""Build a prompt scp by optionally adding random premium speaker prompts."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Tuple


PromptRow = Tuple[str, str]


def parse_base_prompt_line(line: str, source: Path, lineno: int) -> PromptRow:
    line = line.strip()
    if "|" in line:
        wav, text = line.split("|", 1)
        wav = wav.strip()
        text = text.strip()
    else:
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            raise ValueError(f"{source}:{lineno}: expected wav|text or utt wav text")
        _, wav, text = parts
    if not wav or not text:
        raise ValueError(f"{source}:{lineno}: empty wav or text")
    wav_path = Path(wav).expanduser()
    if not wav_path.exists():
        raise FileNotFoundError(f"{source}:{lineno}: wav not found: {wav}")
    return str(wav_path.resolve()), text


def load_base_prompt_scp(path: Path) -> List[PromptRow]:
    rows: List[PromptRow] = []
    with path.open("r", encoding="utf-8") as fin:
        for lineno, raw in enumerate(fin, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(parse_base_prompt_line(line, path, lineno))
    if not rows:
        raise ValueError(f"no valid prompt rows in {path}")
    return rows


def parse_premium_line(line: str, source: Path, lineno: int) -> PromptRow:
    line = line.strip()
    if "|" in line:
        return parse_base_prompt_line(line, source, lineno)
    parts = line.split(maxsplit=2)
    if len(parts) < 3:
        raise ValueError(f"{source}:{lineno}: expected utt wav text")
    _, wav, text = parts
    wav_path = Path(wav).expanduser()
    if not wav_path.exists():
        raise FileNotFoundError(f"{source}:{lineno}: wav not found: {wav}")
    text = text.strip()
    if not text:
        raise ValueError(f"{source}:{lineno}: empty text")
    return str(wav_path.resolve()), text


def load_premium_scp(path: Path) -> List[PromptRow]:
    rows: List[PromptRow] = []
    with path.open("r", encoding="utf-8") as fin:
        for lineno, raw in enumerate(fin, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(parse_premium_line(line, path, lineno))
    if not rows:
        raise ValueError(f"no valid premium rows in {path}")
    return rows


def sample_rows(rows: List[PromptRow], count: int, rng: random.Random, label: str) -> List[PromptRow]:
    if count <= 0:
        return []
    if count > len(rows):
        raise ValueError(f"requested {count} {label} prompts, but only {len(rows)} rows are available")
    return rng.sample(rows, count)


def write_prompt_scp(path: Path, rows: Iterable[PromptRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for wav, text in rows:
            fout.write(f"{wav}|{text}\n")


def write_manifest(path: Path, base_count: int, female_rows: List[PromptRow], male_rows: List[PromptRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        fout.write("group\tidx\tprompt_wav\tprompt_text\n")
        for idx, (wav, text) in enumerate(female_rows, 1):
            fout.write(f"female\t{idx}\t{wav}\t{text}\n")
        for idx, (wav, text) in enumerate(male_rows, 1):
            fout.write(f"male\t{idx}\t{wav}\t{text}\n")
        fout.write(f"base\t{base_count}\t\t\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-prompt-scp", type=Path, required=True)
    parser.add_argument("--female-scp", type=Path, required=True)
    parser.add_argument("--male-scp", type=Path, required=True)
    parser.add_argument("--per-gender", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1986)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--include-base", action="store_true")
    args = parser.parse_args()

    base_rows = load_base_prompt_scp(args.base_prompt_scp)
    female_all = load_premium_scp(args.female_scp)
    male_all = load_premium_scp(args.male_scp)
    rng = random.Random(args.seed)
    female_rows = sample_rows(female_all, args.per_gender, rng, "female")
    male_rows = sample_rows(male_all, args.per_gender, rng, "male")

    output_rows: List[PromptRow] = []
    if args.include_base:
        output_rows.extend(base_rows)
    output_rows.extend(female_rows)
    output_rows.extend(male_rows)
    write_prompt_scp(args.output, output_rows)

    if args.manifest is not None:
        write_manifest(args.manifest, len(base_rows) if args.include_base else 0, female_rows, male_rows)

    print(
        f"wrote {len(output_rows)} prompts to {args.output} "
        f"(base={len(base_rows) if args.include_base else 0}, "
        f"female={len(female_rows)}, male={len(male_rows)}, seed={args.seed})"
    )


if __name__ == "__main__":
    main()
