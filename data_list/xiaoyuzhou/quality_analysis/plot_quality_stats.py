#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


METRIC_KEYS = [
    "mos_score",
    "snr_score",
    "spk_similar_minimum_score",
    "quality_score",
]

GRADE_ORDER = ["A_plus", "A", "B", "C", "D"]


def safe_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def percentile(arr, p):
    if len(arr) == 0:
        return None
    return float(np.percentile(np.array(arr, dtype=np.float64), p))


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_summary_csv(path: Path, metric_stats: dict):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "count", "mean", "std", "min", "p50", "p90", "p95", "p99", "max"])
        for k, s in metric_stats.items():
            w.writerow([
                k, s["count"], s["mean"], s["std"], s["min"], s["p50"], s["p90"], s["p95"], s["p99"], s["max"]
            ])


def plot_histograms(metric_values: dict, out_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, k in enumerate(METRIC_KEYS):
        vals = metric_values[k]
        ax = axes[i]
        if len(vals) == 0:
            ax.set_title(f"{k} (no data)")
            continue
        ax.hist(vals, bins=80, alpha=0.85)
        ax.set_title(f"Histogram: {k}")
        ax.set_xlabel(k)
        ax.set_ylabel("count")

    plt.tight_layout()
    plt.savefig(out_dir / "histograms_metrics.png", dpi=180)
    plt.close()


def plot_boxplot(metric_values: dict, out_dir: Path):
    labels = []
    data = []
    for k in METRIC_KEYS:
        if len(metric_values[k]) > 0:
            labels.append(k)
            data.append(metric_values[k])

    if not data:
        return

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title("Boxplot of Quality Metrics")
    plt.ylabel("value")
    plt.tight_layout()
    plt.savefig(out_dir / "boxplot_metrics.png", dpi=180)
    plt.close()


def plot_grade_distribution(grade_counter: Counter, out_dir: Path):
    labels = [g for g in GRADE_ORDER if grade_counter[g] > 0]
    counts = [grade_counter[g] for g in labels]

    if not labels:
        return

    # Bar
    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts)
    plt.title("Quality Grade Distribution (Bar)")
    plt.xlabel("quality_grade")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "grade_distribution_bar.png", dpi=180)
    plt.close()

    # Pie
    plt.figure(figsize=(7, 7))
    plt.pie(counts, labels=labels, autopct="%.2f%%", startangle=90)
    plt.title("Quality Grade Distribution (Pie)")
    plt.tight_layout()
    plt.savefig(out_dir / "grade_distribution_pie.png", dpi=180)
    plt.close()


def plot_top_speakers(speaker_counter: Counter, out_dir: Path, top_n=20):
    top = speaker_counter.most_common(top_n)
    if not top:
        return

    speakers = [x[0] for x in top]
    counts = [x[1] for x in top]

    plt.figure(figsize=(14, 6))
    plt.bar(range(len(speakers)), counts)
    plt.xticks(range(len(speakers)), speakers, rotation=45, ha="right")
    plt.title(f"Top {top_n} Speakers by Count")
    plt.xlabel("speaker")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / f"top_{top_n}_speakers.png", dpi=180)
    plt.close()


def plot_scatter(sample_rows: list, out_dir: Path):
    if len(sample_rows) == 0:
        return

    mos = np.array([r["mos_score"] for r in sample_rows], dtype=np.float64)
    snr = np.array([r["snr_score"] for r in sample_rows], dtype=np.float64)
    spk = np.array([r["spk_similar_minimum_score"] for r in sample_rows], dtype=np.float64)
    quality = np.array([r["quality_score"] for r in sample_rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(mos, quality, s=6, alpha=0.25)
    axes[0].set_xlabel("mos_score")
    axes[0].set_ylabel("quality_score")
    axes[0].set_title("MOS vs Quality")

    axes[1].scatter(snr, quality, s=6, alpha=0.25)
    axes[1].set_xlabel("snr_score")
    axes[1].set_ylabel("quality_score")
    axes[1].set_title("SNR vs Quality")

    axes[2].scatter(spk, quality, s=6, alpha=0.25)
    axes[2].set_xlabel("spk_similar_minimum_score")
    axes[2].set_ylabel("quality_score")
    axes[2].set_title("SPK Similarity vs Quality")

    plt.tight_layout()
    plt.savefig(out_dir / "scatter_vs_quality.png", dpi=180)
    plt.close()


def plot_correlation_heatmap(sample_rows: list, out_dir: Path):
    if len(sample_rows) < 2:
        return

    mat = np.array(
        [[r["mos_score"], r["snr_score"], r["spk_similar_minimum_score"], r["quality_score"]] for r in sample_rows],
        dtype=np.float64,
    )
    corr = np.corrcoef(mat, rowvar=False)
    labels = ["mos_score", "snr_score", "spk_similar_minimum_score", "quality_score"]

    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title("Correlation Matrix")
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            plt.text(j, i, f"{corr[i, j]:.3f}", ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_matrix.png", dpi=180)
    plt.close()

    return corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Input JSONL path",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--sample_for_scatter",
        type=int,
        default=50000,
        help="Reservoir sample size for scatter/correlation plots",
    )
    parser.add_argument(
        "--top_speaker_n",
        type=int,
        default=20,
        help="Top N speakers for speaker bar chart",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260422,
        help="Random seed",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    in_path = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    metric_values = {k: [] for k in METRIC_KEYS}
    grade_counter = Counter()
    speaker_counter = Counter()

    total_lines = 0
    valid_rows = 0
    bad_json = 0
    missing_core = 0

    # reservoir sample for scatter/correlation
    sample_rows = []
    seen_for_sample = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                bad_json += 1
                continue

            row = {}
            ok = True
            for k in METRIC_KEYS:
                v = safe_float(obj.get(k))
                if v is None:
                    ok = False
                    break
                row[k] = v

            if not ok:
                missing_core += 1
                continue

            valid_rows += 1
            for k in METRIC_KEYS:
                metric_values[k].append(row[k])

            g = obj.get("quality_grade", "UNKNOWN")
            grade_counter[g] += 1

            spk = obj.get("speaker", "UNKNOWN")
            speaker_counter[spk] += 1

            # reservoir sample
            seen_for_sample += 1
            if len(sample_rows) < args.sample_for_scatter:
                sample_rows.append(row)
            else:
                j = random.randint(1, seen_for_sample)
                if j <= args.sample_for_scatter:
                    sample_rows[j - 1] = row

    # numeric summary
    metric_stats = {}
    for k, vals in metric_values.items():
        if len(vals) == 0:
            metric_stats[k] = {
                "count": 0, "mean": None, "std": None, "min": None,
                "p50": None, "p90": None, "p95": None, "p99": None, "max": None
            }
            continue

        arr = np.array(vals, dtype=np.float64)
        metric_stats[k] = {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "p50": percentile(vals, 50),
            "p90": percentile(vals, 90),
            "p95": percentile(vals, 95),
            "p99": percentile(vals, 99),
            "max": float(arr.max()),
        }

    corr = plot_correlation_heatmap(sample_rows, out_dir)
    plot_histograms(metric_values, out_dir)
    plot_boxplot(metric_values, out_dir)
    plot_grade_distribution(grade_counter, out_dir)
    plot_top_speakers(speaker_counter, out_dir, top_n=args.top_speaker_n)
    plot_scatter(sample_rows, out_dir)

    summary = {
        "input_jsonl": str(in_path),
        "total_lines": total_lines,
        "valid_rows": valid_rows,
        "bad_json": bad_json,
        "missing_core_metrics": missing_core,
        "sample_for_scatter": len(sample_rows),
        "metric_stats": metric_stats,
        "grade_counts": dict(grade_counter),
        "top_speakers": speaker_counter.most_common(args.top_speaker_n),
        "correlation_matrix_order": ["mos_score", "snr_score", "spk_similar_minimum_score", "quality_score"],
        "correlation_matrix": corr.tolist() if corr is not None else None,
    }

    save_json(out_dir / "quality_stats_summary.json", summary)
    save_summary_csv(out_dir / "quality_stats_summary.csv", metric_stats)

    print("[DONE] Statistics and plots generated.")
    print(f"[OUT ] {out_dir}")
    print("Generated files:")
    print("  - quality_stats_summary.json")
    print("  - quality_stats_summary.csv")
    print("  - histograms_metrics.png")
    print("  - boxplot_metrics.png")
    print("  - grade_distribution_bar.png")
    print("  - grade_distribution_pie.png")
    print(f"  - top_{args.top_speaker_n}_speakers.png")
    print("  - scatter_vs_quality.png")
    print("  - correlation_matrix.png")


if __name__ == "__main__":
    main()