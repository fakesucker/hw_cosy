#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRIC_KEYS = [
    "mos_score",
    "snr_score",
    "spk_similar_minimum_score",
    "quality_score",
]

GRADE_ORDER = ["A_plus", "A", "B", "C", "D"]


class RunningStats:
    def __init__(self):
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = None
        self.max = None

    def add(self, x: float):
        self.n += 1
        self.sum += x
        self.sumsq += x * x
        if self.min is None or x < self.min:
            self.min = x
        if self.max is None or x > self.max:
            self.max = x

    def summary(self):
        if self.n == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        mean = self.sum / self.n
        var = max(0.0, self.sumsq / self.n - mean * mean)
        std = math.sqrt(var)
        return {
            "count": int(self.n),
            "mean": float(mean),
            "std": float(std),
            "min": float(self.min),
            "max": float(self.max),
        }


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


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def default_bins():
    # Fixed bins for stable plots on very large corpora.
    return {
        "mos_score": np.linspace(0.0, 5.0, 101),
        "snr_score": np.linspace(-10.0, 60.0, 141),
        "spk_similar_minimum_score": np.linspace(0.0, 1.0, 101),
        "quality_score": np.linspace(0.0, 1.0, 101),
    }


def hist_index(edges, x):
    # Returns bin index in [0, len(edges)-2], clamped.
    i = int(np.searchsorted(edges, x, side="right")) - 1
    if i < 0:
        return 0
    if i >= len(edges) - 1:
        return len(edges) - 2
    return i


def approx_percentile_from_hist(edges, counts, p):
    total = int(np.sum(counts))
    if total == 0:
        return None
    target = total * p / 100.0
    cumsum = np.cumsum(counts)
    idx = int(np.searchsorted(cumsum, target, side="left"))
    if idx < 0:
        idx = 0
    if idx >= len(counts):
        idx = len(counts) - 1
    return float((edges[idx] + edges[idx + 1]) / 2.0)


def plot_histograms(edges_map, counts_map, out_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, k in enumerate(METRIC_KEYS):
        ax = axes[i]
        edges = edges_map[k]
        counts = counts_map[k]
        centers = (edges[:-1] + edges[1:]) / 2.0
        widths = np.diff(edges)
        ax.bar(centers, counts, width=widths, alpha=0.85, align="center")
        ax.set_title(f"Histogram: {k}")
        ax.set_xlabel(k)
        ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "histograms_metrics.png", dpi=180)
    plt.close()


def plot_grade_distribution(grade_counter: Counter, out_dir: Path):
    labels = [g for g in GRADE_ORDER if grade_counter[g] > 0]
    counts = [grade_counter[g] for g in labels]
    if not labels:
        return
    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts)
    plt.title("Quality Grade Distribution (Bar)")
    plt.xlabel("quality_grade")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "grade_distribution_bar.png", dpi=180)
    plt.close()

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


def plot_scatter(sample_rows, out_dir: Path):
    if not sample_rows:
        return None
    mos = np.array([r["mos_score"] for r in sample_rows], dtype=np.float64)
    snr = np.array([r["snr_score"] for r in sample_rows], dtype=np.float64)
    spk = np.array([r["spk_similar_minimum_score"] for r in sample_rows], dtype=np.float64)
    quality = np.array([r["quality_score"] for r in sample_rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(mos, quality, s=5, alpha=0.2)
    axes[0].set_xlabel("mos_score")
    axes[0].set_ylabel("quality_score")
    axes[0].set_title("MOS vs Quality")

    axes[1].scatter(snr, quality, s=5, alpha=0.2)
    axes[1].set_xlabel("snr_score")
    axes[1].set_ylabel("quality_score")
    axes[1].set_title("SNR vs Quality")

    axes[2].scatter(spk, quality, s=5, alpha=0.2)
    axes[2].set_xlabel("spk_similar_minimum_score")
    axes[2].set_ylabel("quality_score")
    axes[2].set_title("SPK Similarity vs Quality")

    plt.tight_layout()
    plt.savefig(out_dir / "scatter_vs_quality.png", dpi=180)
    plt.close()

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
    plt.title("Correlation Matrix (Reservoir Sample)")
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            plt.text(j, i, f"{corr[i, j]:.3f}", ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_matrix.png", dpi=180)
    plt.close()
    return corr


def write_summary_csv(path: Path, metric_stats):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "count", "mean", "std", "min", "p50_approx", "p90_approx", "p95_approx", "p99_approx", "max"])
        for k, s in metric_stats.items():
            w.writerow([k, s["count"], s["mean"], s["std"], s["min"], s["p50"], s["p90"], s["p95"], s["p99"], s["max"]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sample_for_scatter", type=int, default=50000)
    parser.add_argument("--top_speaker_n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260422)
    args = parser.parse_args()

    random.seed(args.seed)

    in_path = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    bins = default_bins()
    hist_counts = {k: np.zeros(len(bins[k]) - 1, dtype=np.int64) for k in METRIC_KEYS}
    stats = {k: RunningStats() for k in METRIC_KEYS}
    grade_counter = Counter()
    speaker_counter = Counter()

    total_lines = 0
    valid_rows = 0
    bad_json = 0
    missing_core = 0

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
                x = row[k]
                stats[k].add(x)
                idx = hist_index(bins[k], x)
                hist_counts[k][idx] += 1

            grade_counter[obj.get("quality_grade", "UNKNOWN")] += 1
            speaker_counter[obj.get("speaker", "UNKNOWN")] += 1

            seen_for_sample += 1
            if len(sample_rows) < args.sample_for_scatter:
                sample_rows.append(row)
            else:
                j = random.randint(1, seen_for_sample)
                if j <= args.sample_for_scatter:
                    sample_rows[j - 1] = row

    metric_stats = {}
    for k in METRIC_KEYS:
        base = stats[k].summary()
        base["p50"] = approx_percentile_from_hist(bins[k], hist_counts[k], 50)
        base["p90"] = approx_percentile_from_hist(bins[k], hist_counts[k], 90)
        base["p95"] = approx_percentile_from_hist(bins[k], hist_counts[k], 95)
        base["p99"] = approx_percentile_from_hist(bins[k], hist_counts[k], 99)
        metric_stats[k] = base

    corr = plot_scatter(sample_rows, out_dir)
    plot_histograms(bins, hist_counts, out_dir)
    plot_grade_distribution(grade_counter, out_dir)
    plot_top_speakers(speaker_counter, out_dir, top_n=args.top_speaker_n)

    summary = {
        "input_jsonl": str(in_path),
        "total_lines": total_lines,
        "valid_rows": valid_rows,
        "bad_json": bad_json,
        "missing_core_metrics": missing_core,
        "sample_for_scatter": len(sample_rows),
        "metric_stats_approx_percentile": metric_stats,
        "grade_counts": dict(grade_counter),
        "top_speakers": speaker_counter.most_common(args.top_speaker_n),
        "correlation_matrix_order": ["mos_score", "snr_score", "spk_similar_minimum_score", "quality_score"],
        "correlation_matrix": corr.tolist() if corr is not None else None,
        "note": "Percentiles are approximated from fixed-width histograms for memory efficiency on very large JSONL.",
    }

    save_json(out_dir / "quality_stats_summary.json", summary)
    write_summary_csv(out_dir / "quality_stats_summary.csv", metric_stats)

    print("[DONE] Large-file statistics and plots generated.")
    print(f"[OUT ] {out_dir}")


if __name__ == "__main__":
    main()
