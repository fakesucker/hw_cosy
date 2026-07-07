#!/usr/bin/env python3
# Copyright (c) 2026
#
# Compare baseline and OPD rows from midterm CER/latency summaries.

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def read_tsv(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fin:
        return list(csv.DictReader(fin, delimiter="\t"))


def parse_float(value: str):
    value = (value or "").strip()
    if not value or value == "NA":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value) -> str:
    if value is None:
        return ""
    return "{:.6f}".format(float(value))


def role_for_exp(exp: str, baseline_prefix: str, opd_prefix: str) -> str:
    if exp.startswith(baseline_prefix):
        return "baseline"
    if exp.startswith(opd_prefix):
        return "opd"
    return ""


def latency_map(rows: List[dict]) -> Dict[Tuple[str, str, str], dict]:
    result = {}
    for row in rows:
        result[(row.get("meta", ""), row.get("exp", ""), row.get("epoch", ""))] = row
    return result


def enrich(row: dict, latencies: Dict[Tuple[str, str, str], dict]) -> dict:
    key = (row.get("meta", ""), row.get("exp", ""), row.get("epoch", ""))
    latency = latencies.get(key, {})
    enriched = dict(row)
    enriched["cer"] = parse_float(row.get("cer_pct", ""))
    enriched["rtf"] = parse_float(latency.get("rtf_done_over_audio", ""))
    enriched["done_mean"] = parse_float(latency.get("done_mean", ""))
    enriched["first_audio_mean"] = parse_float(latency.get("first_audio_mean", ""))
    return enriched


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare OPD vs baseline CER and latency summaries")
    parser.add_argument("--summary-all", required=True, help="summary_all.tsv from run_opd_midterm_eval.sh")
    parser.add_argument("--latency-summary", default="", help="summary_latency.tsv")
    parser.add_argument("--output", required=True, help="Output comparison TSV")
    parser.add_argument("--baseline-prefix", default="baseline_")
    parser.add_argument("--opd-prefix", default="opd_")
    args = parser.parse_args()

    summary_rows = read_tsv(Path(args.summary_all))
    latency_rows = read_tsv(Path(args.latency_summary)) if args.latency_summary else []
    latencies = latency_map(latency_rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    grouped: Dict[Tuple[str, str], Dict[str, List[dict]]] = {}
    for row in summary_rows:
        role = role_for_exp(row.get("exp", ""), args.baseline_prefix, args.opd_prefix)
        if not role:
            continue
        key = (row.get("set", ""), row.get("meta", ""))
        grouped.setdefault(key, {"baseline": [], "opd": []})[role].append(enrich(row, latencies))

    header = [
        "set",
        "meta",
        "baseline_exp",
        "baseline_epoch",
        "opd_exp",
        "opd_epoch",
        "baseline_status",
        "opd_status",
        "baseline_cer_pct",
        "opd_cer_pct",
        "cer_delta_opd_minus_base",
        "cer_relative_improvement_pct",
        "baseline_rtf",
        "opd_rtf",
        "rtf_delta_opd_minus_base",
        "baseline_done_mean",
        "opd_done_mean",
        "done_delta_opd_minus_base",
        "baseline_first_audio_mean",
        "opd_first_audio_mean",
        "first_audio_delta_opd_minus_base",
    ]

    out_rows = []
    for key in sorted(grouped):
        base_rows = grouped[key]["baseline"]
        opd_rows = grouped[key]["opd"]
        for base in base_rows:
            for opd in opd_rows:
                cer_delta = None
                cer_rel = None
                if base["cer"] is not None and opd["cer"] is not None:
                    cer_delta = opd["cer"] - base["cer"]
                    if base["cer"] != 0:
                        cer_rel = (base["cer"] - opd["cer"]) / base["cer"] * 100.0
                rtf_delta = None
                if base["rtf"] is not None and opd["rtf"] is not None:
                    rtf_delta = opd["rtf"] - base["rtf"]
                done_delta = None
                if base["done_mean"] is not None and opd["done_mean"] is not None:
                    done_delta = opd["done_mean"] - base["done_mean"]
                first_audio_delta = None
                if base["first_audio_mean"] is not None and opd["first_audio_mean"] is not None:
                    first_audio_delta = opd["first_audio_mean"] - base["first_audio_mean"]
                out_rows.append([
                    key[0],
                    key[1],
                    base.get("exp", ""),
                    base.get("epoch", ""),
                    opd.get("exp", ""),
                    opd.get("epoch", ""),
                    base.get("status", ""),
                    opd.get("status", ""),
                    fmt(base["cer"]),
                    fmt(opd["cer"]),
                    fmt(cer_delta),
                    fmt(cer_rel),
                    fmt(base["rtf"]),
                    fmt(opd["rtf"]),
                    fmt(rtf_delta),
                    fmt(base["done_mean"]),
                    fmt(opd["done_mean"]),
                    fmt(done_delta),
                    fmt(base["first_audio_mean"]),
                    fmt(opd["first_audio_mean"]),
                    fmt(first_audio_delta),
                ])

    with output.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow(header)
        writer.writerows(out_rows)

    print("opd_eval_comparison_tsv={}".format(output))
    print("opd_eval_comparison_rows={}".format(len(out_rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
