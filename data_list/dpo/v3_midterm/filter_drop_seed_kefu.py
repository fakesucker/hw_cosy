#!/usr/bin/env python3
"""Remove JSONL records whose utt starts with seed_kefu."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_jsonl",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "speech_tokens.jsonl",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output path (default: overwrite input after backup)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="do not create .bak before overwrite",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        default=True,
        help="overwrite input (default)",
    )
    args = parser.parse_args()

    inp = args.input_jsonl.resolve()
    if not inp.is_file():
        print(f"[ERROR] not found: {inp}", file=sys.stderr)
        sys.exit(1)

    out = (args.output or inp).resolve()
    tmp = out.with_suffix(out.suffix + ".tmp")

    total = kept = dropped = 0
    with inp.open(encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            total += 1
            obj = json.loads(line)
            utt = obj.get("utt", "")
            if str(utt).startswith("seed_kefu"):
                dropped += 1
                continue
            fout.write(line + "\n")
            kept += 1

    if out == inp and not args.no_backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = inp.with_suffix(inp.suffix + f".bak_{ts}")
        shutil.copy2(inp, bak)
        print(f"backup -> {bak}", file=sys.stderr)

    tmp.replace(out)
    print(
        f"done: total={total} dropped(seed_kefu)={dropped} kept={kept} -> {out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
