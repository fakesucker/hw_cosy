#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Group DPO JSONL rows by utt prefix.

Input row example:
{
  "utt": "05783_027",
  "token": [...],
  "text": "...",
  "wavpath": "/path/a.wav"
}

Grouping rule:
- If utt ends with "_000" ~ "_999", use prefix before the last "_NNN" as group_id.
- Otherwise use full utt as group_id.

Outputs:
1) grouped JSONL (one line per group)
2) flat JSONL with extra fields: group_id / group_size / group_rank
3) TSV stats for quick filtering
"""

import argparse
import json
import re
from collections import defaultdict, Counter
from pathlib import Path


UTT_SUFFIX_RE = re.compile(r"^(.*)_([0-9]{3})$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_jsonl", required=True, help="Input DPO jsonl path")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument(
        "--min_group_size",
        type=int,
        default=2,
        help="Keep groups whose size >= min_group_size",
    )
    p.add_argument(
        "--max_group_size",
        type=int,
        default=0,
        help="Keep groups whose size <= max_group_size, 0 means no upper limit",
    )
    return p.parse_args()


def split_utt(utt: str):
    """
    Returns:
      group_id: prefix for clustering
      suffix_idx: int if endswith _NNN else -1
    """
    if not isinstance(utt, str):
        return str(utt), -1
    m = UTT_SUFFIX_RE.match(utt)
    if not m:
        return utt, -1
    prefix, suffix = m.group(1), m.group(2)
    return prefix, int(suffix)


def normalize_row(obj: dict):
    # unify wav path key name
    if "wav_path" not in obj and "wavpath" in obj:
        obj["wav_path"] = obj["wavpath"]
    return obj


def main():
    args = parse_args()
    in_path = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    bad_json = 0
    no_utt = 0

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

            obj = normalize_row(obj)
            utt = obj.get("utt")
            if utt is None:
                no_utt += 1
                continue

            group_id, suffix_idx = split_utt(utt)
            obj["_group_id"] = group_id
            obj["_suffix_idx"] = suffix_idx
            obj["_line_no"] = line_no
            groups[group_id].append(obj)

    # sort each group by suffix_idx first, fallback by utt then original line
    for gid in groups:
        groups[gid].sort(
            key=lambda x: (
                10**9 if x["_suffix_idx"] < 0 else x["_suffix_idx"],
                str(x.get("utt", "")),
                x["_line_no"],
            )
        )

    # apply group size filter
    kept = {}
    for gid, items in groups.items():
        n = len(items)
        if n < args.min_group_size:
            continue
        if args.max_group_size > 0 and n > args.max_group_size:
            continue
        kept[gid] = items

    grouped_jsonl = out_dir / "dpo_grouped_by_prefix.jsonl"
    flat_with_group_jsonl = out_dir / "dpo_rows_with_group.jsonl"
    stats_tsv = out_dir / "dpo_group_stats.tsv"
    summary_json = out_dir / "dpo_group_summary.json"

    # 1) grouped jsonl
    with grouped_jsonl.open("w", encoding="utf-8") as w:
        for gid in sorted(kept.keys()):
            items = kept[gid]
            texts = [x.get("text") for x in items if x.get("text") is not None]
            text_counter = Counter(texts)
            top_text, top_text_count = ("", 0)
            if text_counter:
                top_text, top_text_count = text_counter.most_common(1)[0]

            row = {
                "group_id": gid,
                "group_size": len(items),
                "unique_text_count": len(text_counter),
                "majority_text": top_text,
                "majority_text_count": top_text_count,
                "items": [
                    {
                        "utt": x.get("utt"),
                        "suffix_idx": x.get("_suffix_idx"),
                        "text": x.get("text"),
                        "wav_path": x.get("wav_path"),
                        "token": x.get("token"),
                    }
                    for x in items
                ],
            }
            w.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 2) flat jsonl (row-wise, easy for later scoring/filtering)
    with flat_with_group_jsonl.open("w", encoding="utf-8") as w:
        for gid in sorted(kept.keys()):
            items = kept[gid]
            group_size = len(items)
            for rank, x in enumerate(items):
                out = {k: v for k, v in x.items() if not k.startswith("_")}
                out["group_id"] = gid
                out["group_size"] = group_size
                out["group_rank"] = rank
                out["suffix_idx"] = x.get("_suffix_idx")
                w.write(json.dumps(out, ensure_ascii=False) + "\n")

    # 3) stats tsv
    with stats_tsv.open("w", encoding="utf-8") as w:
        w.write("group_id\tgroup_size\tunique_text_count\tmajority_text_count\tmajority_text_ratio\n")
        for gid in sorted(kept.keys()):
            items = kept[gid]
            texts = [x.get("text") for x in items if x.get("text") is not None]
            text_counter = Counter(texts)
            unique_text_count = len(text_counter)
            majority_text_count = text_counter.most_common(1)[0][1] if text_counter else 0
            ratio = (majority_text_count / len(items)) if items else 0.0
            w.write(
                f"{gid}\t{len(items)}\t{unique_text_count}\t{majority_text_count}\t{ratio:.6f}\n"
            )

    group_sizes = [len(v) for v in kept.values()]
    size_counter = Counter(group_sizes)

    summary = {
        "input_jsonl": str(in_path),
        "total_groups_raw": len(groups),
        "total_groups_kept": len(kept),
        "total_rows_raw": sum(len(v) for v in groups.values()),
        "total_rows_kept": sum(len(v) for v in kept.values()),
        "bad_json": bad_json,
        "missing_utt": no_utt,
        "min_group_size": args.min_group_size,
        "max_group_size": args.max_group_size,
        "group_size_distribution": dict(sorted(size_counter.items())),
        "outputs": {
            "grouped_jsonl": str(grouped_jsonl),
            "flat_with_group_jsonl": str(flat_with_group_jsonl),
            "stats_tsv": str(stats_tsv),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
