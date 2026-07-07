#!/usr/bin/env python3
# Copyright (c) 2026
#
# Aggregate latency.tsv files produced by infer_seed.py for OPD midterm eval.

import argparse
import csv
from pathlib import Path
from typing import Iterable, List


NUMERIC_FIELDS = [
    "first_chunk_latency",
    "first_audio_latency",
    "utterance_done_latency",
    "audio_duration_sec",
]


def read_float(value: str):
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def percentile(values: List[float], pct: float) -> str:
    if not values:
        return ""
    ordered = sorted(values)
    if len(ordered) == 1:
        return "{:.6f}".format(ordered[0])
    pos = (len(ordered) - 1) * pct
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    frac = pos - lower
    value = ordered[lower] * (1.0 - frac) + ordered[upper] * frac
    return "{:.6f}".format(value)


def mean(values: List[float]) -> str:
    if not values:
        return ""
    return "{:.6f}".format(sum(values) / len(values))


def iter_latency_files(infer_root: Path) -> Iterable[Path]:
    if not infer_root.is_dir():
        return []
    return sorted(infer_root.glob("*/*/epoch_*_whole/latency.tsv"))


def summarize_latency_file(path: Path, infer_root: Path) -> List[str]:
    rel = path.relative_to(infer_root)
    meta = rel.parts[0]
    exp = rel.parts[1]
    epoch = rel.parts[2]
    rows = []
    with path.open(newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        for row in reader:
            rows.append(row)

    status_ok = [row for row in rows if row.get("status") == "ok"]
    status_fail = [row for row in rows if row.get("status") != "ok"]
    values = {
        field: [
            parsed for parsed in (read_float(row.get(field, "")) for row in status_ok)
            if parsed is not None
        ]
        for field in NUMERIC_FIELDS
    }
    duration_sum = sum(values["audio_duration_sec"])
    done_sum = sum(values["utterance_done_latency"])
    rtf = ""
    if duration_sum > 0 and done_sum > 0:
        rtf = "{:.6f}".format(done_sum / duration_sum)

    return [
        meta,
        exp,
        epoch,
        str(len(rows)),
        str(len(status_ok)),
        str(len(status_fail)),
        mean(values["first_chunk_latency"]),
        percentile(values["first_chunk_latency"], 0.50),
        percentile(values["first_chunk_latency"], 0.95),
        mean(values["first_audio_latency"]),
        percentile(values["first_audio_latency"], 0.50),
        percentile(values["first_audio_latency"], 0.95),
        mean(values["utterance_done_latency"]),
        percentile(values["utterance_done_latency"], 0.50),
        percentile(values["utterance_done_latency"], 0.95),
        mean(values["audio_duration_sec"]),
        rtf,
        str(path),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize OPD midterm latency TSV files")
    parser.add_argument("--infer-root", required=True, help="OPD eval infer root, usually OUTPUT_BASE/infer")
    parser.add_argument("--output", required=True, help="Output summary_latency.tsv path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    infer_root = Path(args.infer_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "meta",
        "exp",
        "epoch",
        "rows",
        "ok",
        "fail",
        "first_chunk_mean",
        "first_chunk_p50",
        "first_chunk_p95",
        "first_audio_mean",
        "first_audio_p50",
        "first_audio_p95",
        "done_mean",
        "done_p50",
        "done_p95",
        "audio_duration_mean",
        "rtf_done_over_audio",
        "latency_tsv",
    ]
    rows = [summarize_latency_file(path, infer_root) for path in iter_latency_files(infer_root)]
    with output.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)
    print("latency_summary_tsv={}".format(output))
    print("latency_summary_rows={}".format(len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
