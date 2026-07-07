#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sample grouped DPO JSONL for Gemini labeling.

Pipeline:
1) Keep groups whose item count is exactly `--required_group_size`
2) Keep groups whose token sequences are all different within group
3) Randomly sample N groups
4) For each sampled group, randomly sample K items
5) Save sampled grouped JSONL + summary
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Sample grouped DPO data for Gemini labeling")
    p.add_argument("--input_jsonl", required=True, help="Grouped jsonl path")
    p.add_argument(
        "--out_dir",
        default="",
        help="Output directory. If empty, auto create beside input jsonl.",
    )
    p.add_argument("--sample_groups", type=int, default=3000, help="Number of groups to sample")
    p.add_argument("--sample_items_per_group", type=int, default=20, help="Items to sample per group")
    p.add_argument("--required_group_size", type=int, default=30, help="Keep groups with this exact size")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def token_fingerprint(token) -> str:
    s = json.dumps(token, ensure_ascii=False, separators=(",", ":"))
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def is_group_all_tokens_unique(items: list[dict], required_group_size: int) -> bool:
    if len(items) != required_group_size:
        return False
    seen = set()
    for it in items:
        tok = it.get("token")
        if tok is None:
            return False
        fp = token_fingerprint(tok)
        if fp in seen:
            return False
        seen.add(fp)
    return True


def build_sampled_group(src_group: dict, sampled_items: list[dict]) -> dict:
    texts = [x.get("text") for x in sampled_items if x.get("text") is not None]
    text_counter = Counter(texts)
    majority_text, majority_text_count = ("", 0)
    if text_counter:
        majority_text, majority_text_count = text_counter.most_common(1)[0]

    sampled_items_sorted = sorted(
        sampled_items,
        key=lambda x: (
            int(x.get("suffix_idx")) if str(x.get("suffix_idx", "")).isdigit() else 10**9,
            str(x.get("utt", "")),
        ),
    )
    for rank, it in enumerate(sampled_items_sorted):
        it["group_rank"] = rank

    out = {
        "group_id": src_group.get("group_id"),
        "group_size": len(sampled_items_sorted),
        "unique_text_count": len(text_counter),
        "majority_text": majority_text,
        "majority_text_count": majority_text_count,
        "items": sampled_items_sorted,
        "sampling_meta": {
            "source_group_size": src_group.get("group_size", len(src_group.get("items", []))),
            "sampled_items_per_group": len(sampled_items_sorted),
        },
    }
    return out


def main():
    args = parse_args()
    in_path = Path(args.input_jsonl).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"input_jsonl not found: {in_path}")

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = in_path.parent / f"{in_path.stem}_sampled_for_gemini_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "grouped_sampled_for_gemini.jsonl"
    out_summary = out_dir / "grouped_sampled_for_gemini_summary.json"

    random.seed(args.seed)

    total_groups = 0
    bad_json = 0
    invalid_items = 0
    wrong_group_size = 0
    non_unique_token_groups = 0
    eligible_groups: list[dict] = []

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            total_groups += 1
            try:
                obj = json.loads(s)
            except Exception:
                bad_json += 1
                continue

            items = obj.get("items")
            if not isinstance(items, list):
                invalid_items += 1
                continue
            if len(items) != args.required_group_size:
                wrong_group_size += 1
                continue
            if not is_group_all_tokens_unique(items, args.required_group_size):
                non_unique_token_groups += 1
                continue
            eligible_groups.append(obj)

    requested_groups = args.sample_groups
    sampled_group_count = min(requested_groups, len(eligible_groups))
    sampled_groups = random.sample(eligible_groups, sampled_group_count)

    with out_jsonl.open("w", encoding="utf-8") as w:
        for g in sampled_groups:
            gid = g.get("group_id")
            items = g.get("items", [])
            if len(items) < args.sample_items_per_group:
                continue
            sampled_items = random.sample(items, args.sample_items_per_group)
            sampled_obj = build_sampled_group(g, sampled_items)
            if gid is not None:
                sampled_obj["group_id"] = gid
            w.write(json.dumps(sampled_obj, ensure_ascii=False) + "\n")

    summary = {
        "input_jsonl": str(in_path),
        "output_jsonl": str(out_jsonl),
        "total_groups": total_groups,
        "bad_json": bad_json,
        "invalid_items_field": invalid_items,
        "filtered_wrong_group_size": wrong_group_size,
        "filtered_non_unique_token_groups": non_unique_token_groups,
        "eligible_groups": len(eligible_groups),
        "requested_sample_groups": requested_groups,
        "actual_sampled_groups": sampled_group_count,
        "sample_items_per_group": args.sample_items_per_group,
        "required_group_size": args.required_group_size,
        "seed": args.seed,
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

