#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List

# ====== 配置 ======
INPUT_JSONL = Path("/home/work_nfs23/xmren/tokens/extracted_sentences_score.jsonl")
OUT_DIR = Path("/home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/xiaoyuzhou/quality_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 综合打分权重（可按你的偏好调）
W_MOS = 0.45
W_SNR = 0.25
W_SPK = 0.30

# snr 归一化区间（经验值，可调）
SNR_MIN = 10.0
SNR_MAX = 50.0

# 按 quality_score 的分级阈值
# quality_score in [0, 1]
GRADE_THRESHOLDS = [
    ("A_plus", 0.90),
    ("A", 0.82),
    ("B", 0.72),
    ("C", 0.60),
    ("D", 0.00),
]

# 输出 top 样本数量
TOP_K = 500000  # 可改大/小

# 直方图桶
HIST_BINS = {
    "mos_score": [2.5, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 5.0],
    "snr_score": [0, 10, 20, 25, 30, 35, 40, 45, 50, 999],
    "spk_similar_minimum_score": [0.0, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01],
    "quality_score": [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01],
}
# ==================


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def norm_snr(snr: float) -> float:
    return clip((snr - SNR_MIN) / (SNR_MAX - SNR_MIN), 0.0, 1.0)


def compute_quality(mos: float, snr: float, spk: float) -> float:
    # mos 通常在 [1,5]，先映射到 [0,1]
    mos_norm = clip((mos - 1.0) / 4.0, 0.0, 1.0)
    snr_norm = norm_snr(snr)
    spk_norm = clip(spk, 0.0, 1.0)
    q = W_MOS * mos_norm + W_SNR * snr_norm + W_SPK * spk_norm
    return float(clip(q, 0.0, 1.0))


def assign_grade(q: float) -> str:
    for g, th in GRADE_THRESHOLDS:
        if q >= th:
            return g
    return "D"


class RunningStats:
    """Welford 算法：流式均值/方差"""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_v = math.inf
        self.max_v = -math.inf
        self.samples = []  # 为了分位数（如太大可改 reservoir）

    def add(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        self.min_v = min(self.min_v, x)
        self.max_v = max(self.max_v, x)
        self.samples.append(x)

    def summary(self) -> Dict[str, Any]:
        if self.n == 0:
            return {}
        var = self.M2 / self.n if self.n > 0 else 0.0
        std = math.sqrt(var)
        s = sorted(self.samples)
        def pct(p: float):
            if not s:
                return None
            idx = int((len(s) - 1) * p)
            return s[idx]
        return {
            "count": self.n,
            "mean": self.mean,
            "std": std,
            "min": self.min_v,
            "p01": pct(0.01),
            "p05": pct(0.05),
            "p10": pct(0.10),
            "p25": pct(0.25),
            "p50": pct(0.50),
            "p75": pct(0.75),
            "p90": pct(0.90),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": self.max_v,
        }


def hist_bucket(x: float, edges: List[float]) -> str:
    # edges: [a, b, c, ...] -> [a,b), [b,c), ...
    for i in range(len(edges) - 1):
        if edges[i] <= x < edges[i + 1]:
            return f"[{edges[i]}, {edges[i+1]})"
    return f"[{edges[-2]}, {edges[-1]}]"


def main():
    stats = {
        "mos_score": RunningStats(),
        "snr_score": RunningStats(),
        "spk_similar_minimum_score": RunningStats(),
        "quality_score": RunningStats(),
    }
    grade_counter = Counter()
    speaker_counter = Counter()
    bad_lines = 0
    missing_fields = 0
    total = 0

    # 直方图
    hists = {k: Counter() for k in HIST_BINS.keys()}

    # 分级输出文件
    grade_files = {
        g: (OUT_DIR / f"grade_{g}.jsonl").open("w", encoding="utf-8")
        for g, _ in GRADE_THRESHOLDS
    }

    # top K（简化：先全收再排序，磁盘大可接受；如内存不足可改最小堆）
    top_items = []

    with INPUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            total += 1
            try:
                obj = json.loads(s)
            except Exception:
                bad_lines += 1
                continue

            if not all(k in obj for k in ["mos_score", "snr_score", "spk_similar_minimum_score"]):
                missing_fields += 1
                continue

            try:
                mos = float(obj["mos_score"])
                snr = float(obj["snr_score"])
                spk = float(obj["spk_similar_minimum_score"])
            except Exception:
                missing_fields += 1
                continue

            q = compute_quality(mos, snr, spk)
            grade = assign_grade(q)

            obj["quality_score"] = q
            obj["quality_grade"] = grade

            stats["mos_score"].add(mos)
            stats["snr_score"].add(snr)
            stats["spk_similar_minimum_score"].add(spk)
            stats["quality_score"].add(q)

            grade_counter[grade] += 1
            if "speaker" in obj:
                speaker_counter[obj["speaker"]] += 1

            hists["mos_score"][hist_bucket(mos, HIST_BINS["mos_score"])] += 1
            hists["snr_score"][hist_bucket(snr, HIST_BINS["snr_score"])] += 1
            hists["spk_similar_minimum_score"][hist_bucket(spk, HIST_BINS["spk_similar_minimum_score"])] += 1
            hists["quality_score"][hist_bucket(q, HIST_BINS["quality_score"])] += 1

            grade_files[grade].write(json.dumps(obj, ensure_ascii=False) + "\n")
            top_items.append((q, obj))

    for fp in grade_files.values():
        fp.close()

    # 导出 top K
    top_items.sort(key=lambda x: x[0], reverse=True)
    top_k = top_items[:TOP_K]
    top_path = OUT_DIR / f"top_{TOP_K}_quality.jsonl"
    with top_path.open("w", encoding="utf-8") as w:
        for _, obj in top_k:
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # 也导出一个严格高质量集合（可直接用于训练）
    high_path = OUT_DIR / "high_quality_strict.jsonl"
    with high_path.open("w", encoding="utf-8") as w:
        for q, obj in top_items:
            if (
                obj["mos_score"] >= 3.6 and
                obj["snr_score"] >= 30 and
                obj["spk_similar_minimum_score"] >= 0.85 and
                q >= 0.82
            ):
                w.write(json.dumps(obj, ensure_ascii=False) + "\n")

    report = {
        "input_jsonl": str(INPUT_JSONL),
        "total_lines_seen": total,
        "bad_json_lines": bad_lines,
        "missing_or_invalid_metric_lines": missing_fields,
        "valid_lines": stats["quality_score"].n,
        "weights": {"mos": W_MOS, "snr": W_SNR, "spk_sim": W_SPK},
        "snr_norm_range": [SNR_MIN, SNR_MAX],
        "metric_summary": {k: v.summary() for k, v in stats.items()},
        "grade_counts": dict(grade_counter),
        "speaker_top20": speaker_counter.most_common(20),
        "histograms": {k: dict(v) for k, v in hists.items()},
        "outputs": {
            "grade_files": {g: str(OUT_DIR / f"grade_{g}.jsonl") for g, _ in GRADE_THRESHOLDS},
            "top_k_file": str(top_path),
            "high_quality_strict": str(high_path),
        },
    }

    report_path = OUT_DIR / "quality_report.json"
    with report_path.open("w", encoding="utf-8") as w:
        json.dump(report, w, ensure_ascii=False, indent=2)

    print(f"Done. report: {report_path}")
    print(f"Top file: {top_path}")
    print(f"Strict high-quality: {high_path}")


if __name__ == "__main__":
    main()