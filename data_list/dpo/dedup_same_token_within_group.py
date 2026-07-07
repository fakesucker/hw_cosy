#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deduplicate rows with identical token sequence within each group_id.

Input JSONL row example (from dpo_rows_with_group.jsonl):
{
  "utt": "...",
  "token": [...],
  "text": "...",
  "wav_path": "...",
  "group_id": "...",
  "group_size": 30,
  "group_rank": 0,
  "suffix_idx": 0
}

Rule:
- Within each group_id, rows with exactly identical token list are duplicates.
- Keep only the first occurrence (stable by file order).
"""

import argparse
import hashlib
import json
from collections import defaultdict, Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_jsonl", required=True, help="Input JSONL path with group_id field")
    p.add_argument("--output_jsonl", required=True, help="Output deduplicated JSONL path")
    p.add_argument("--summary_json", required=True, help="Output summary JSON path")
    p.add_argument("--drop_no_group", action="store_true", help="Drop rows without group_id")
    return p.parse_args()


def token_fingerprint(token):
    # robust hash for long token arrays
    s = json.dumps(token, ensure_ascii=False, separators=(",", ":"))
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def main():
    args = parse_args()
    in_path = Path(args.input_jsonl)
    out_path = Path(args.output_jsonl)
    sum_path = Path(args.summary_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = defaultdict(set)  # group_id -> set(token_hash)
    group_in = Counter()
    group_out = Counter()

    total_in = 0
    total_out = 0
    removed = 0
    bad_json = 0
    no_group = 0
    no_token = 0

    with in_path.open("r", encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as w:
        for line in f:
            s = line.strip()
            if not s:
                continue
            total_in += 1
            try:
                obj = json.loads(s)
            except Exception:
                bad_json += 1
                continue

            gid = obj.get("group_id")
            if gid is None:
                no_group += 1
                if args.drop_no_group:
                    continue
                gid = "__NO_GROUP__"

            tok = obj.get("token")
            if tok is None:
                no_token += 1
                # no token -> keep by default, but count under group
                group_in[gid] += 1
                group_out[gid] += 1
                w.write(json.dumps(obj, ensure_ascii=False) + "\n")
                total_out += 1
                continue

            fp = token_fingerprint(tok)
            group_in[gid] += 1
            if fp in seen[gid]:
                removed += 1
                continue
            seen[gid].add(fp)
            group_out[gid] += 1
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")
            total_out += 1

    # group-level reduction stats
    reduced_groups = 0
    max_removed_in_group = 0
    for gid, in_n in group_in.items():
        out_n = group_out.get(gid, 0)
        if out_n < in_n:
            reduced_groups += 1
            if in_n - out_n > max_removed_in_group:
                max_removed_in_group = in_n - out_n

    summary = {
        "input_jsonl": str(in_path),
        "output_jsonl": str(out_path),
        "total_in": total_in,
        "total_out": total_out,
        "removed_same_token": removed,
        "removed_ratio": (removed / total_in) if total_in else 0.0,
        "bad_json": bad_json,
        "missing_group_id": no_group,
        "missing_token": no_token,
        "groups_total": len(group_in),
        "groups_reduced": reduced_groups,
        "max_removed_in_single_group": max_removed_in_group,
    }
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
