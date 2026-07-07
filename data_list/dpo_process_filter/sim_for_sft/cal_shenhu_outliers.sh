#!/usr/bin/env bash
#
# =============================================================================
# shenhu.jsonl：WavLM 说话人向量 -> PCA -> KMeans -> 离群点 + 聚类统计 + t-SNE 图
# =============================================================================
#
# 【本机默认路径】无参数运行时直接使用下列路径（可按需改脚本内常量）：
#   JSONL   /home/work_nfs23/xmren/data/shenhu/shenhu.jsonl
#   权重文件 /home/work_nfs23/xmren/data/shenhu/seed-tts-eval/wavlm_large_finetune.pth
#            （与脚本同目录；若在 weights/ 下请三参数指定或改 DEFAULT_CKPT）
#   输出目录 /home/work_nfs23/xmren/data/shenhu/shenhu_spk_outlier_run
#
# 【用法】
#   1) 无参（用上面默认路径）：
#        bash cal_shenhu_outliers.sh
#
#   2) 显式传参：
#        bash cal_shenhu_outliers.sh <jsonl> <输出目录> <wavlm_large_finetune.pth>
#
# 【复制示例】
#   cd /home/work_nfs23/xmren/data/shenhu/seed-tts-eval
#   bash cal_shenhu_outliers.sh
#
#   或指定输出目录与权重文件：
#   bash cal_shenhu_outliers.sh \
#     /home/work_nfs23/xmren/data/shenhu/shenhu.jsonl \
#     /home/work_nfs23/xmren/data/shenhu/shenhu_spk_outlier_run \
#     /home/work_nfs23/xmren/data/shenhu/seed-tts-eval/wavlm_large_finetune.pth
#
# 【可选环境变量】
#   DEVICE      默认 cuda:0，例：DEVICE=cuda:1
#   EXTRA_ARGS  传给 python 的额外参数，例：
#               EXTRA_ARGS='--limit 500'                    # 只跑前 500 条
#               EXTRA_ARGS='--reuse_cache --outlier_top_frac 0.02'   # 复用 embeddings.npz
#               EXTRA_ARGS='--reuse_cache --do_tsne --limit 2000'    # 画 t-SNE（建议先限量）
#   SV_DIR      可选，显式指定 speaker_verification 目录（包含 verification.py）
#               例：SV_DIR=/home/work_nfs23/xmren/data/shenhu/seed-tts-eval/thirdparty/UniSpeech/downstreams/speaker_verification
#
# 输出见：
#   <输出目录>/shenhu_spk_outliers.txt
#   <输出目录>/cluster_stats.tsv
#   <输出目录>/points_with_cluster.tsv
#   <输出目录>/tsne_points.tsv            (仅 --do_tsne)
#   <输出目录>/tsne_clusters.png         (仅 --do_tsne)
#   <输出目录>/embeddings.npz            (向量缓存)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 与上文「本机默认路径」一致；换机器时请改这三项或改用三参数调用。
DEFAULT_JSONL="/home/work_nfs23/xmren/data/shenhu/shenhu.jsonl"
# 权重与 cal_shenhu_outliers.sh 同目录（当前为 seed-tts-eval/wavlm_large_finetune.pth）
DEFAULT_CKPT="${SCRIPT_DIR}/wavlm_large_finetune.pth"
DEFAULT_OUT_DIR="/home/work_nfs23/xmren/data/shenhu/shenhu_spk_outlier_run"

usage() {
  cat <<'USAGE'
用法:
  bash cal_shenhu_outliers.sh
      无参：使用脚本内 DEFAULT_* 路径（本机 jsonl / 权重 / 输出目录）

  bash cal_shenhu_outliers.sh <jsonl> <输出目录> <wavlm_large_finetune.pth>

环境变量: DEVICE=cuda:0  EXTRA_ARGS='--limit 200' 等
USAGE
}

if [ "${#}" -eq 0 ]; then
  JSONL="${DEFAULT_JSONL}"
  OUT_DIR="${DEFAULT_OUT_DIR}"
  CKPT="${DEFAULT_CKPT}"
elif [ "${#}" -eq 3 ]; then
  JSONL="${1}"
  OUT_DIR="${2}"
  CKPT="${3}"
else
  usage
  echo ""
  echo "错误: 需要 0 个参数（走默认路径）或 3 个参数 <jsonl> <输出目录> <checkpoint.pth>，当前 ${#} 个。"
  exit 1
fi

if [ ! -f "${JSONL}" ]; then
  echo "错误: 找不到 jsonl: ${JSONL}"
  exit 1
fi
if [ ! -f "${CKPT}" ]; then
  echo "错误: 找不到权重文件: ${CKPT}"
  echo "      请将 wavlm_large_finetune.pth 放在脚本同目录: ${SCRIPT_DIR}/ 或传第三参数指定路径"
  exit 1
fi

mkdir -p "${OUT_DIR}"

python3 "${SCRIPT_DIR}/shenhu_jsonl_wavlm_outliers.py" \
  --jsonl "${JSONL}" \
  --checkpoint "${CKPT}" \
  --out_dir "${OUT_DIR}" \
  --device "${DEVICE:-cuda:0}" \
  --sv_dir "${SV_DIR:-}" \
  ${EXTRA_ARGS:-}
