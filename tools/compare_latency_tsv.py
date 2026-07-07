#!/usr/bin/env python3
import argparse
import csv
import os
import statistics
from typing import Dict, Iterable, List, Optional, Tuple


NUMERIC_FIELDS = [
    "text_chunk_count",
    "audio_yield_count",
    "first_chunk_latency",
    "first_audio_latency",
    "utterance_done_latency",
    "audio_duration_sec",
]

META_FIELDS = [
    "infer_mode",
    "text_input_mode",
    "stream_audio",
    "wav_path",
    "error",
]


def _read_tsv(path: str) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or "utt" not in reader.fieldnames:
            raise ValueError(f"Invalid latency TSV, missing utt column: {path}")
        for row in reader:
            utt = (row.get("utt") or "").strip()
            if not utt:
                continue
            rows[utt] = row
    return rows


def _as_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    return float(s)


def _fmt_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _safe_mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return statistics.mean(values)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _default_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".tsv"):
        base = base[:-4]
    return base or "latency"


def _build_detail_row(
    utt: str,
    left: Optional[Dict[str, str]],
    right: Optional[Dict[str, str]],
    left_name: str,
    right_name: str,
) -> Dict[str, str]:
    row: Dict[str, str] = {"utt": utt}
    left_status = (left or {}).get("status", "")
    right_status = (right or {}).get("status", "")
    row["left_present"] = "1" if left is not None else "0"
    row["right_present"] = "1" if right is not None else "0"
    row[f"{left_name}_status"] = left_status
    row[f"{right_name}_status"] = right_status
    row["paired_ok"] = "1" if left_status == "ok" and right_status == "ok" else "0"

    for field in META_FIELDS:
        row[f"{left_name}_{field}"] = (left or {}).get(field, "")
        row[f"{right_name}_{field}"] = (right or {}).get(field, "")

    for field in NUMERIC_FIELDS:
        left_v = _as_float((left or {}).get(field, ""))
        right_v = _as_float((right or {}).get(field, ""))
        row[f"{left_name}_{field}"] = _fmt_float(left_v)
        row[f"{right_name}_{field}"] = _fmt_float(right_v)
        diff = None if left_v is None or right_v is None else right_v - left_v
        row[f"diff_{field}_{right_name}_minus_{left_name}"] = _fmt_float(diff)
    return row


def _write_tsv(path: str, fieldnames: List[str], rows: Iterable[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summary_rows(
    left_rows: Dict[str, Dict[str, str]],
    right_rows: Dict[str, Dict[str, str]],
    detail_rows: List[Dict[str, str]],
    left_name: str,
    right_name: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    union_utts = {row["utt"] for row in detail_rows}
    common_utts = set(left_rows) & set(right_rows)
    left_ok = sum(1 for r in left_rows.values() if r.get("status") == "ok")
    right_ok = sum(1 for r in right_rows.values() if r.get("status") == "ok")
    paired_ok_rows = [r for r in detail_rows if r["paired_ok"] == "1"]

    rows.append({
        "section": "counts",
        "metric": "utt_count",
        "left_mean": str(len(left_rows)),
        "right_mean": str(len(right_rows)),
        "diff_mean": str(len(right_rows) - len(left_rows)),
        "left_name": left_name,
        "right_name": right_name,
        "paired_count": str(len(common_utts)),
        "paired_ok_count": str(len(paired_ok_rows)),
        "notes": f"union={len(union_utts)} left_ok={left_ok} right_ok={right_ok}",
    })

    for field in NUMERIC_FIELDS:
        left_vals = [
            _as_float(row.get(f"{left_name}_{field}", ""))
            for row in paired_ok_rows
        ]
        left_vals = [x for x in left_vals if x is not None]
        right_vals = [
            _as_float(row.get(f"{right_name}_{field}", ""))
            for row in paired_ok_rows
        ]
        right_vals = [x for x in right_vals if x is not None]
        diff_vals = [
            _as_float(row.get(f"diff_{field}_{right_name}_minus_{left_name}", ""))
            for row in paired_ok_rows
        ]
        diff_vals = [x for x in diff_vals if x is not None]
        rows.append({
            "section": "mean_paired_ok",
            "metric": field,
            "left_mean": _fmt_float(_safe_mean(left_vals)),
            "right_mean": _fmt_float(_safe_mean(right_vals)),
            "diff_mean": _fmt_float(_safe_mean(diff_vals)),
            "left_name": left_name,
            "right_name": right_name,
            "paired_count": str(len(common_utts)),
            "paired_ok_count": str(len(paired_ok_rows)),
            "notes": "",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two latency.tsv files by utt")
    parser.add_argument("--left", required=True, help="Baseline / left latency.tsv")
    parser.add_argument("--right", required=True, help="Experiment / right latency.tsv")
    parser.add_argument("--left_name", default="", help="Optional short label for left file")
    parser.add_argument("--right_name", default="", help="Optional short label for right file")
    parser.add_argument("--output_dir", default="", help="Directory to write detail.tsv and summary.tsv")
    args = parser.parse_args()

    left_name = args.left_name or _default_name_from_path(args.left)
    right_name = args.right_name or _default_name_from_path(args.right)
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.right)),
        f"compare_{left_name}_vs_{right_name}",
    )
    _ensure_dir(output_dir)

    left_rows = _read_tsv(args.left)
    right_rows = _read_tsv(args.right)
    all_utts = sorted(set(left_rows) | set(right_rows))
    detail_rows = [
        _build_detail_row(utt, left_rows.get(utt), right_rows.get(utt), left_name, right_name)
        for utt in all_utts
    ]

    detail_fields = [
        "utt",
        "left_present",
        "right_present",
        f"{left_name}_status",
        f"{right_name}_status",
        "paired_ok",
    ]
    for field in META_FIELDS:
        detail_fields.append(f"{left_name}_{field}")
        detail_fields.append(f"{right_name}_{field}")
    for field in NUMERIC_FIELDS:
        detail_fields.append(f"{left_name}_{field}")
        detail_fields.append(f"{right_name}_{field}")
        detail_fields.append(f"diff_{field}_{right_name}_minus_{left_name}")

    detail_path = os.path.join(output_dir, "detail.tsv")
    _write_tsv(detail_path, detail_fields, detail_rows)

    summary_rows = _summary_rows(left_rows, right_rows, detail_rows, left_name, right_name)
    summary_fields = [
        "section",
        "metric",
        "left_name",
        "right_name",
        "left_mean",
        "right_mean",
        "diff_mean",
        "paired_count",
        "paired_ok_count",
        "notes",
    ]
    summary_path = os.path.join(output_dir, "summary.tsv")
    _write_tsv(summary_path, summary_fields, summary_rows)

    print(f"Left: {args.left}")
    print(f"Right: {args.right}")
    print(f"Detail TSV: {detail_path}")
    print(f"Summary TSV: {summary_path}")
    for row in summary_rows:
        if row["section"] != "mean_paired_ok":
            continue
        print(
            f"{row['metric']}: "
            f"{left_name}={row['left_mean']} "
            f"{right_name}={row['right_mean']} "
            f"diff={row['diff_mean']}"
        )


if __name__ == "__main__":
    main()
