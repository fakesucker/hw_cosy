#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-click DPO JSONL preparation pipeline:
1) group by utt prefix
2) deduplicate same token within each group
3) regroup deduplicated rows back to grouped JSONL

Usage:
  python run_dpo_prepare_pipeline.py --input_jsonl xxx.jsonl
  python run_dpo_prepare_pipeline.py --input_jsonl xxx.jsonl --out_dir ./out_xxx
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="One-click DPO clean/dedup/group pipeline")
    p.add_argument("--input_jsonl", required=True, help="Input jsonl path")
    p.add_argument(
        "--out_dir",
        default="",
        help="Output directory. If empty, auto create near input.",
    )
    p.add_argument("--min_group_size", type=int, default=2)
    p.add_argument("--max_group_size", type=int, default=0)
    p.add_argument(
        "--drop_no_group",
        action="store_true",
        help="Drop rows without group_id in dedup/regroup stages",
    )
    return p.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl).resolve()
    if not input_jsonl.exists():
        raise FileNotFoundError(f"input_jsonl not found: {input_jsonl}")

    script_dir = Path(__file__).resolve().parent

    group_script = script_dir / "group_dpo_by_utt_prefix.py"
    dedup_script = script_dir / "dedup_same_token_within_group.py"
    regroup_script = script_dir / "regroup_dedup_rows_to_grouped.py"
    for p in [group_script, dedup_script, regroup_script]:
        if not p.exists():
            raise FileNotFoundError(f"missing required script: {p}")

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = input_jsonl.parent / f"{input_jsonl.stem}_dpo_prepared_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1_dir = out_dir / "01_grouped"
    stage2_dir = out_dir / "02_dedup"
    stage3_dir = out_dir / "03_regrouped"
    for d in [stage1_dir, stage2_dir, stage3_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Stage 1: group by utt prefix
    run_cmd(
        [
            sys.executable,
            str(group_script),
            "--input_jsonl",
            str(input_jsonl),
            "--out_dir",
            str(stage1_dir),
            "--min_group_size",
            str(args.min_group_size),
            "--max_group_size",
            str(args.max_group_size),
        ]
    )

    stage1_flat = stage1_dir / "dpo_rows_with_group.jsonl"
    stage2_flat = stage2_dir / "dpo_rows_with_group_token_dedup.jsonl"
    stage2_summary = stage2_dir / "dpo_rows_with_group_token_dedup_summary.json"

    # Stage 2: dedup same token inside each group
    dedup_cmd = [
        sys.executable,
        str(dedup_script),
        "--input_jsonl",
        str(stage1_flat),
        "--output_jsonl",
        str(stage2_flat),
        "--summary_json",
        str(stage2_summary),
    ]
    if args.drop_no_group:
        dedup_cmd.append("--drop_no_group")
    run_cmd(dedup_cmd)

    stage3_grouped = stage3_dir / "dpo_grouped_by_prefix_token_dedup.jsonl"
    stage3_summary = stage3_dir / "dpo_grouped_by_prefix_token_dedup_summary.json"

    # Stage 3: regroup deduplicated rows
    regroup_cmd = [
        sys.executable,
        str(regroup_script),
        "--input_jsonl",
        str(stage2_flat),
        "--output_jsonl",
        str(stage3_grouped),
        "--summary_json",
        str(stage3_summary),
    ]
    if args.drop_no_group:
        regroup_cmd.append("--drop_no_group")
    run_cmd(regroup_cmd)

    pipeline_summary = {
        "input_jsonl": str(input_jsonl),
        "out_dir": str(out_dir),
        "stage_outputs": {
            "stage1_grouped_jsonl": str(stage1_dir / "dpo_grouped_by_prefix.jsonl"),
            "stage1_flat_jsonl": str(stage1_flat),
            "stage1_stats_tsv": str(stage1_dir / "dpo_group_stats.tsv"),
            "stage1_summary_json": str(stage1_dir / "dpo_group_summary.json"),
            "stage2_dedup_flat_jsonl": str(stage2_flat),
            "stage2_summary_json": str(stage2_summary),
            "stage3_grouped_dedup_jsonl": str(stage3_grouped),
            "stage3_summary_json": str(stage3_summary),
        },
        "args": {
            "min_group_size": args.min_group_size,
            "max_group_size": args.max_group_size,
            "drop_no_group": args.drop_no_group,
        },
    }
    pipeline_summary_path = out_dir / "pipeline_summary.json"
    pipeline_summary_path.write_text(
        json.dumps(pipeline_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nDone. Pipeline outputs:")
    print(json.dumps(pipeline_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

