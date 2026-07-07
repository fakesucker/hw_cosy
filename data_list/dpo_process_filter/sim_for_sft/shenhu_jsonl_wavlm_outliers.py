#!/usr/bin/env python3
"""
Read shenhu.jsonl, extract WavLM-large (SV finetune) speaker embeddings per wav_path,
then StandardScaler + PCA defines an orthonormal coordinate system, KMeans clustering,
and outliers are samples farthest from their assigned cluster centroid (global tail).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import librosa
import matplotlib
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torchaudio.transforms import Resample
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_SEED = 42

def _speaker_verification_path(checkpoint: str, sv_dir: str = "") -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint)) if checkpoint else ""
    candidates = []

    if sv_dir:
        sv_dir_abs = os.path.abspath(sv_dir)
        candidates.append(sv_dir_abs)
        candidates.append(os.path.join(sv_dir_abs, "speaker_verification"))

    candidates.append(os.path.join(root, "thirdparty", "UniSpeech", "downstreams", "speaker_verification"))
    if ckpt_dir:
        candidates.append(os.path.join(ckpt_dir, "thirdparty", "UniSpeech", "downstreams", "speaker_verification"))

    for p in candidates:
        if os.path.isfile(os.path.join(p, "verification.py")):
            return p

    raise FileNotFoundError(
        "找不到 verification.py。请传 --sv_dir 指向包含 verification.py 的目录，"
        f"或保证其位于脚本目录/或checkpoint目录下的 thirdparty/UniSpeech/downstreams/speaker_verification。"
        f" 尝试路径: {candidates}"
    )

def load_wav_tensor(wav_path: str, device: torch.device) -> torch.Tensor:
    wav, sr = librosa.load(wav_path, sr=None, mono=False)
    if len(wav.shape) == 2:
        wav = wav[0, :]
    wav = torch.from_numpy(wav).unsqueeze(0).float()
    wav = Resample(orig_freq=sr, new_freq=16000)(wav)
    return wav.to(device)

@torch.no_grad()
def extract_embeddings(
    records: list[dict],
    checkpoint: str,
    device: str,
    sv_dir: str = "",
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    sv = _speaker_verification_path(checkpoint=checkpoint, sv_dir=sv_dir)
    if sv not in sys.path:
        sys.path.insert(0, sv)
    from verification import init_model  # noqa: E402

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        dev = torch.device(device)
    else:
        dev = torch.device("cpu")
    model = init_model("wavlm_large", checkpoint)
    model.eval()
    if dev.type == "cuda":
        model = model.cuda(dev)

    embs: list[np.ndarray] = []
    utts: list[str] = []
    paths: list[str] = []
    skips: list[str] = []

    for row in tqdm(records, desc="WavLM embed"):
        utt = str(row.get("utt", ""))
        wp = row.get("wav_path", "")
        if not wp or not isinstance(wp, str):
            skips.append(f"{utt}\tmissing_wav_path")
            continue
        if not os.path.isfile(wp):
            skips.append(f"{utt}\tmissing_file\t{wp}")
            continue
        try:
            wav = load_wav_tensor(wp, dev)
            emb = model(wav)
            vec = emb.squeeze(0).detach().float().cpu().numpy()
            embs.append(vec)
            utts.append(utt)
            paths.append(wp)
        except Exception as e:  # noqa: BLE001
            skips.append(f"{utt}\textract_fail\t{wp}\t{type(e).__name__}: {e}")

    if not embs:
        raise RuntimeError("没有成功提取任何向量，请检查 jsonl 与 wav 路径。")
    return np.stack(embs, axis=0), utts, paths, skips


def cluster_and_outliers(
    X: np.ndarray,
    n_clusters: int,
    pca_dim: int,
    outlier_top_frac: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, KMeans, PCA, StandardScaler, np.ndarray, np.ndarray]:
    n, d = X.shape
    pca_n = min(pca_dim, n - 1, d)
    if pca_n < 2:
        raise ValueError(f"样本过少，无法进行 PCA（n={n}, d={d}）")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=pca_n, random_state=random_state)
    Z = pca.fit_transform(Xs)

    k = max(1, min(n_clusters, n))
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(Z)
    centers = km.cluster_centers_

    dist = np.linalg.norm(Z - centers[labels], axis=1)
    thr = np.quantile(dist, 1.0 - outlier_top_frac)
    outlier_mask = dist >= thr
    return Z, labels, km, pca, scaler, dist, outlier_mask


def save_cluster_stats(
    out_path: str,
    labels: np.ndarray,
    dist: np.ndarray,
    outlier_mask: np.ndarray,
) -> None:
    n = len(labels)
    uniq = sorted(set(int(x) for x in labels.tolist()))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "cluster_id\tsize\tfrac\tmean_dist\tmedian_dist\tp95_dist\tmax_dist\toutlier_count\toutlier_frac\n"
        )
        for cid in uniq:
            idx = np.where(labels == cid)[0]
            cdist = dist[idx]
            cout = int(outlier_mask[idx].sum())
            size = len(idx)
            f.write(
                f"{cid}\t{size}\t{size / max(1, n):.6f}\t{float(np.mean(cdist)):.6f}\t"
                f"{float(np.median(cdist)):.6f}\t{float(np.quantile(cdist, 0.95)):.6f}\t"
                f"{float(np.max(cdist)):.6f}\t{cout}\t{cout / max(1, size):.6f}\n"
            )


def save_point_table(
    out_path: str,
    utts: list[str],
    paths: list[str],
    labels: np.ndarray,
    dist: np.ndarray,
    outlier_mask: np.ndarray,
    Z: np.ndarray,
) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "utt",
                "wav_path",
                "cluster_id",
                "dist_to_cluster_center",
                "is_outlier",
                "pca_0",
                "pca_1",
                "pca_2",
            ]
        )
        for i in range(len(utts)):
            z2 = float(Z[i, 2]) if Z.shape[1] > 2 else 0.0
            w.writerow(
                [
                    utts[i],
                    paths[i],
                    int(labels[i]),
                    f"{float(dist[i]):.6f}",
                    int(outlier_mask[i]),
                    f"{float(Z[i, 0]):.6f}",
                    f"{float(Z[i, 1]):.6f}",
                    f"{z2:.6f}",
                ]
            )


def run_tsne(
    Z: np.ndarray,
    random_state: int,
    perplexity: float,
    learning_rate: float,
    max_iter: int,
) -> np.ndarray:
    n = Z.shape[0]
    if n < 5:
        raise ValueError(f"样本过少，无法稳定计算 t-SNE（n={n}）")
    p = min(float(perplexity), float(n - 1))
    if p < 2.0:
        p = 2.0
    tsne = TSNE(
        n_components=2,
        perplexity=p,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=random_state,
        init="pca",
    )
    return tsne.fit_transform(Z).astype(np.float32)


def save_tsne_points(
    out_path: str,
    utts: list[str],
    paths: list[str],
    labels: np.ndarray,
    dist: np.ndarray,
    outlier_mask: np.ndarray,
    tsne_xy: np.ndarray,
) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["utt", "wav_path", "cluster_id", "dist_to_cluster_center", "is_outlier", "tsne_x", "tsne_y"])
        for i in range(len(utts)):
            w.writerow(
                [
                    utts[i],
                    paths[i],
                    int(labels[i]),
                    f"{float(dist[i]):.6f}",
                    int(outlier_mask[i]),
                    f"{float(tsne_xy[i, 0]):.6f}",
                    f"{float(tsne_xy[i, 1]):.6f}",
                ]
            )


def save_tsne_plot(
    out_png: str,
    tsne_xy: np.ndarray,
    utts: list[str],
    labels: np.ndarray,
    dist: np.ndarray,
    outlier_mask: np.ndarray,
    title: str,
    single_cluster_style: bool = False,
    annotate_topn: int = 0,
) -> None:
    fig = plt.figure(figsize=(12, 10), dpi=140)
    ax = fig.add_subplot(111)

    if single_cluster_style:
        ax.scatter(
            tsne_xy[:, 0],
            tsne_xy[:, 1],
            c="#7c8aa0",
            s=11,
            alpha=0.62,
            linewidths=0,
        )
    else:
        sc = ax.scatter(
            tsne_xy[:, 0],
            tsne_xy[:, 1],
            c=labels.astype(np.int32),
            s=10,
            alpha=0.70,
            cmap="tab20",
            linewidths=0,
        )

    oidx = np.where(outlier_mask)[0]
    if len(oidx) > 0:
        ax.scatter(
            tsne_xy[oidx, 0],
            tsne_xy[oidx, 1],
            s=38,
            facecolors="none",
            edgecolors="red",
            linewidths=1.1,
            label=f"outliers ({len(oidx)})",
        )

        if annotate_topn > 0:
            sorted_idx = sorted(
                [int(i) for i in oidx.tolist()],
                key=lambda i: float(dist[i]),
                reverse=True,
            )[:annotate_topn]
            for i in sorted_idx:
                ax.annotate(
                    utts[i],
                    (float(tsne_xy[i, 0]), float(tsne_xy[i, 1])),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color="darkred",
                    alpha=0.9,
                )
        ax.legend(loc="best")
    ax.set_title(title)
    ax.set_xlabel("t-SNE x")
    ax.set_ylabel("t-SNE y")
    if not single_cluster_style:
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("cluster id")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="shenhu.jsonl 路径")
    ap.add_argument("--checkpoint", required=True, help="wavlm_large_finetune.pth")
    ap.add_argument("--out_dir", required=True, help="输出目录（写 txt / npz / skip 日志）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--reuse_cache",
        action="store_true",
        help="若存在 embeddings.npz 则跳过 WavLM 前向，只做 PCA+聚类+离群",
    )
    ap.add_argument("--n_clusters", type=int, default=32, help="KMeans 簇数（可设为 1，表示全局单簇中心）")
    ap.add_argument(
        "--single_cluster_mode",
        action="store_true",
        help="强制单簇模式：把全部样本当作一个簇，用全局中心距离筛离群点",
    )
    ap.add_argument("--pca_dim", type=int, default=64, help="PCA 主成分数上限")
    ap.add_argument(
        "--outlier_top_frac",
        type=float,
        default=0.01,
        help="按「到所属簇中心距离」全局分位数取离群：取距离最大的该比例（如 0.01=1%）",
    )
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 条 jsonl（0 表示全量，用于试跑）")
    ap.add_argument(
        "--sv_dir",
        default="",
        help="speaker_verification 目录（包含 verification.py），不传则自动按脚本目录和checkpoint目录回退查找",
    )
    ap.add_argument("--do_tsne", action="store_true", help="是否计算 t-SNE 并写出 tsne 点表和 png 图")
    ap.add_argument("--tsne_perplexity", type=float, default=30.0)
    ap.add_argument("--tsne_learning_rate", type=float, default=200.0)
    ap.add_argument("--tsne_max_iter", type=int, default=1000)
    ap.add_argument(
        "--tsne_single_cluster_style",
        action="store_true",
        help="t-SNE 图使用统一底色（不按 cluster_id 着色），适合单簇场景",
    )
    ap.add_argument(
        "--tsne_annotate_topn",
        type=int,
        default=0,
        help="在 t-SNE 图上标注距离最大的前 N 个离群点 utt（0=不标注）",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache_npz = os.path.join(args.out_dir, "embeddings.npz")
    out_txt = os.path.join(args.out_dir, "shenhu_spk_outliers.txt")
    skip_txt = os.path.join(args.out_dir, "shenhu_spk_embed_skips.txt")
    cluster_stat_tsv = os.path.join(args.out_dir, "cluster_stats.tsv")
    points_tsv = os.path.join(args.out_dir, "points_with_cluster.tsv")
    tsne_points_tsv = os.path.join(args.out_dir, "tsne_points.tsv")
    tsne_png = os.path.join(args.out_dir, "tsne_clusters.png")

    records: list[dict] = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if args.limit and args.limit > 0:
        records = records[: args.limit]

    skips: list[str] = []
    utts: list[str] = []
    paths: list[str] = []
    if args.reuse_cache and os.path.isfile(cache_npz):
        data = np.load(cache_npz, allow_pickle=True)
        X = data["emb"].astype(np.float32)
        utts = data["utt"].tolist()
        paths = data["wav_path"].tolist()
    else:
        X, utts, paths, skips = extract_embeddings(
            records,
            args.checkpoint,
            args.device,
            sv_dir=args.sv_dir,
        )
        np.savez_compressed(
            cache_npz,
            emb=X.astype(np.float32),
            utt=np.array(utts, dtype=object),
            wav_path=np.array(paths, dtype=object),
        )

    if skips:
        with open(skip_txt, "w", encoding="utf-8") as f:
            f.write("utt\treason\textra\n")
            for s in skips:
                parts = s.split("\t")
                f.write("\t".join(parts) + "\n")

    effective_k = 1 if args.single_cluster_mode else args.n_clusters
    Z, labels, km, pca, _scaler, dist, outlier_mask = cluster_and_outliers(
        X,
        n_clusters=effective_k,
        pca_dim=args.pca_dim,
        outlier_top_frac=args.outlier_top_frac,
        random_state=_SEED,
    )

    thr = float(np.quantile(dist, 1.0 - args.outlier_top_frac))
    n_out = int(outlier_mask.sum())
    evr = float(np.sum(pca.explained_variance_ratio_))

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(
            "# 流程: per-utt WavLM-large(SV) embedding -> StandardScaler -> PCA 正交坐标"
            f" -> KMeans(k={effective_k}) -> 离群=到所属簇中心距离 >= 全局 {100 * (1 - args.outlier_top_frac):.4g}% 分位\n"
        )
        k_eff = getattr(km, "n_clusters_", km.n_clusters)
        f.write(f"# n_samples={len(utts)} n_outliers={n_out} pca_dim={pca.n_components_} kmeans_k={k_eff}\n")
        f.write(f"# pca_explained_variance_ratio_sum={evr:.6f} dist_threshold={thr:.6f}\n")
        f.write(
            "utt\twav_path\tdist_to_cluster_center\tcluster_id\t"
            "pca_0\tpca_1\tpca_2\n"
        )
        for i in np.where(outlier_mask)[0]:
            i = int(i)
            z2 = float(Z[i, 2]) if Z.shape[1] > 2 else 0.0
            f.write(
                f"{utts[i]}\t{paths[i]}\t{dist[i]:.6f}\t{int(labels[i])}\t"
                f"{Z[i, 0]:.6f}\t{Z[i, 1]:.6f}\t{z2:.6f}\n"
            )

    save_cluster_stats(cluster_stat_tsv, labels, dist, outlier_mask)
    save_point_table(points_tsv, utts, paths, labels, dist, outlier_mask, Z)

    if args.do_tsne:
        try:
            tsne_xy = run_tsne(
                Z,
                random_state=_SEED,
                perplexity=args.tsne_perplexity,
                learning_rate=args.tsne_learning_rate,
                max_iter=args.tsne_max_iter,
            )
            save_tsne_points(tsne_points_tsv, utts, paths, labels, dist, outlier_mask, tsne_xy)
            save_tsne_plot(
                tsne_png,
                tsne_xy,
                utts,
                labels,
                dist,
                outlier_mask,
                title=f"WavLM latent t-SNE (n={len(utts)}, outliers={n_out}, k={getattr(km, 'n_clusters_', km.n_clusters)})",
                single_cluster_style=(args.tsne_single_cluster_style or args.single_cluster_mode),
                annotate_topn=max(0, args.tsne_annotate_topn),
            )
            print(f"写入: {tsne_points_tsv}")
            print(f"写入: {tsne_png}")
        except Exception as e:  # noqa: BLE001
            print(f"警告: t-SNE 失败，已跳过。{type(e).__name__}: {e}")

    print(f"写入: {out_txt} (离群 {n_out}/{len(utts)})")
    print(f"写入: {cluster_stat_tsv}")
    print(f"写入: {points_tsv}")
    if skips:
        print(f"跳过/失败明细: {skip_txt} ({len(skips)} 条)")


if __name__ == "__main__":
    main()
