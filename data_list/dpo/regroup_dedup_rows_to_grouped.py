#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Regroup deduplicated flat rows (one row per sample) into one row per group.

Input:
  dpo_rows_with_group_token_dedup.jsonl

Output:
  dpo_grouped_by_prefix_token_dedup.jsonl   # one line per group
  dpo_grouped_by_prefix_token_dedup_summary.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_jsonl", required=True, help="Flat dedup JSONL with group_id")
    p.add_argument("--output_jsonl", required=True, help="Grouped JSONL output path")
    p.add_argument("--summary_json", required=True, help="Summary json output path")
    p.add_argument("--drop_no_group", action="store_true", help="Drop rows without group_id")
    return p.parse_args()


def safe_int(x, default=10**9):
    try:
        return int(x)
    except Exception:
        return default


def main():
    args = parse_args()
    in_path = Path(args.input_jsonl)
    out_path = Path(args.output_jsonl)
    sum_path = Path(args.summary_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    bad_json = 0
    missing_group = 0
    total_rows = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                bad_json += 1
                continue
            total_rows += 1
            gid = obj.get("group_id")
            if gid is None:
                missing_group += 1
                if args.drop_no_group:
                    continue
                gid = "__NO_GROUP__"
            obj["_line_no"] = line_no
            groups[gid].append(obj)

    size_counter = Counter()
    with out_path.open("w", encoding="utf-8") as w:
        for gid in sorted(groups.keys()):
            items = groups[gid]
            items.sort(
                key=lambda x: (
                    safe_int(x.get("suffix_idx"), 10**9),
                    safe_int(x.get("group_rank"), 10**9),
                    str(x.get("utt", "")),
                    x.get("_line_no", 10**9),
                )
            )
            clean_items = []
            texts = []
            for rank, x in enumerate(items):
                y = {k: v for k, v in x.items() if not k.startswith("_")}
                y["group_rank"] = rank
                clean_items.append(
                    {
                        "utt": y.get("utt"),
                        "suffix_idx": y.get("suffix_idx"),
                        "group_rank": y.get("group_rank"),
                        "text": y.get("text"),
                        "wav_path": y.get("wav_path", y.get("wavpath")),
                        "token": y.get("token"),
                    }
                )
                if y.get("text") is not None:
                    texts.append(y.get("text"))

            text_counter = Counter(texts)
            majority_text, majority_text_count = ("", 0)
            if text_counter:
                majority_text, majority_text_count = text_counter.most_common(1)[0]

            group_size = len(clean_items)
            size_counter[group_size] += 1
            out_obj = {
                "group_id": gid,
                "group_size": group_size,
                "unique_text_count": len(text_counter),
                "majority_text": majority_text,
                "majority_text_count": majority_text_count,
                "items": clean_items,
            }
            w.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    summary = {
        "input_jsonl": str(in_path),
        "output_jsonl": str(out_path),
        "total_rows_in": total_rows,
        "total_groups_out": len(groups),
        "bad_json": bad_json,
        "missing_group_id": missing_group,
        "group_size_distribution": dict(sorted(size_counter.items())),
    }
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
