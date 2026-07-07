#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
"""Build blind A/B wav mapping + optional copy, matching data_list/AB-test/mix_samples_mapping.csv schema.

Example:
  python3 build_ab_mix.py \\
    --label-a shenhuonly --wav-dir-a .../shenhuonly/epoch_2_whole \\
    --label-b bigbatch --wav-dir-b .../bigbatch/epoch_4_whole \\
    --csv-out mix_samples_mapping.csv \\
    --copy-dest ./mixed
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import string
from pathlib import Path


def _wav_stems(wav_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {wav_dir}")
    for p in wav_dir.glob("*.wav"):
        stem = p.stem
        if stem in out:
            raise ValueError(f"Duplicate stem {stem} under {wav_dir}")
        out[stem] = p.resolve()
    return out


def _random_suffix(used: set[str], length: int = 6, rng: random.Random | None = None) -> str:
    rng = rng or random.Random()
    alphabet = string.ascii_letters + string.digits
    while True:
        s = "".join(rng.choice(alphabet) for _ in range(length))
        if s not in used:
            used.add(s)
            return s


def main() -> None:
    ap = argparse.ArgumentParser(description="AB-test blind rename: intersect wav stems, random suffix, CSV + optional copy.")
    ap.add_argument("--label-a", required=True, help="Short name for model A (CSV source_model_folder)")
    ap.add_argument("--wav-dir-a", type=Path, required=True, help="Directory containing A side *.wav (e.g. .../epoch_4_whole)")
    ap.add_argument("--label-b", required=True, help="Short name for model B")
    ap.add_argument("--wav-dir-b", type=Path, required=True, help="Directory containing B side *.wav")
    ap.add_argument("--epoch-folder-a", default="", help="CSV source_epoch_folder for A (default: wav-dir-a basename)")
    ap.add_argument("--epoch-folder-b", default="", help="CSV source_epoch_folder for B (default: wav-dir-b basename)")
    ap.add_argument("--csv-out", type=Path, required=True, help="Output mapping CSV path")
    ap.add_argument("--copy-dest", type=Path, default=None, help="If set, copy renamed wavs into this directory (flat)")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible suffixes")
    args = ap.parse_args()

    epoch_a = args.epoch_folder_a or args.wav_dir_a.name
    epoch_b = args.epoch_folder_b or args.wav_dir_b.name

    stems_a = _wav_stems(args.wav_dir_a)
    stems_b = _wav_stems(args.wav_dir_b)
    common = sorted(set(stems_a.keys()) & set(stems_b.keys()))
    if not common:
        raise SystemExit("No common *.wav stems between A and B directories.")

    rng = random.Random(args.seed)
    used_suffixes: set[str] = set()
    rows: list[dict[str, str]] = []

    for stem in common:
        for label, epoch_name, src_path in (
            (args.label_a, epoch_a, stems_a[stem]),
            (args.label_b, epoch_b, stems_b[stem]),
        ):
            sfx = _random_suffix(used_suffixes, rng=rng)
            mixed = f"{stem}_{sfx}.wav"
            rows.append(
                {
                    "mixed_filename": mixed,
                    "suffix": sfx,
                    "source_model_folder": label,
                    "source_epoch_folder": epoch_name,
                    "source_filename": f"{stem}.wav",
                    "source_full_path": str(src_path),
                }
            )

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mixed_filename",
        "suffix",
        "source_model_folder",
        "source_epoch_folder",
        "source_filename",
        "source_full_path",
    ]
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    if args.copy_dest:
        args.copy_dest.mkdir(parents=True, exist_ok=True)
        for r in rows:
            src = Path(r["source_full_path"])
            dst = args.copy_dest / r["mixed_filename"]
            shutil.copy2(src, dst)

    print(f"Common stems: {len(common)}  ->  rows: {len(rows)}  (2 per stem)")
    print(f"Wrote {args.csv_out}")
    if args.copy_dest:
        print(f"Copied to {args.copy_dest}")


if __name__ == "__main__":
    main()
