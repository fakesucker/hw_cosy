#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Randomly sample N lines from a JSONL file.")
    p.add_argument("--input_jsonl", required=True, help="Input jsonl path")
    p.add_argument("--output_jsonl", required=True, help="Output sampled jsonl path")
    p.add_argument("--sample_n", type=int, default=50, help="Number of samples")
    p.add_argument("--seed", type=int, default=20260427, help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input_jsonl)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(s)

    total = len(rows)
    if total == 0:
        raise ValueError(f"Empty input: {in_path}")

    n = min(args.sample_n, total)
    random.seed(args.seed)
    sampled = random.sample(rows, n)

    with out_path.open("w", encoding="utf-8") as w:
        for s in sampled:
            # validate json format
            obj = json.loads(s)
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"input={in_path}")
    print(f"output={out_path}")
    print(f"total={total}")
    print(f"sampled={n}")
    print(f"seed={args.seed}")


if __name__ == "__main__":
    main()
