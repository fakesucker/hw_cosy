#!/usr/bin/env bash
set -euo pipefail

# Non-stream OPSD distillation on dialogue.lst + shenhu_filtered_wo_outliers98
# text/token data. OPSD training is fixed to unistream/full-sequence; pair it
# with whole-text, non-stream eval in run_opsd_step_ckpt_eval.sh.
# Defaults keep the quick OPSD recipe: lr=1e-6 from config, save every 20 steps,
# log every step, and stop after 500 steps. OPSD keeps student and teacher on
# GPU, so 4000 frames is the stable 4x4090 default; override if memory allows.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

export DISTILL_MODE="${DISTILL_MODE:-opsd}"
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-/home/work_nfs23/hkxie/ckpt/huawei/cosyvoice2/dpo_xiaoyuzhou_shenhu_10-5_1e-6_bigbatch/epoch_5_whole.pt}"
export TRAIN_DATA="${TRAIN_DATA:-${SCRIPT_DIR}/data/opsd_dialogue_shenhu_fixed03729/train.data.list}"
export CV_DATA="${CV_DATA:-${SCRIPT_DIR}/data/opsd_dialogue_shenhu_fixed03729/cv.data.list}"
export RUN_NAME="${RUN_NAME:-opd_distill_opsd_topk16_dialogue_shenhu_fixed03729_nonstream_mf4000}"
export CONFIG="${CONFIG:-conf/cosyvoice2_sft_1e-6_spk.yaml}"

export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-500}"
export SAVE_PER_STEP="${SAVE_PER_STEP:-20}"
export LOG_INTERVAL="${LOG_INTERVAL:-1}"
export MAX_FRAMES_IN_BATCH="${MAX_FRAMES_IN_BATCH:-4000}"
export SKIP_CV_ON_STEP_SAVE="${SKIP_CV_ON_STEP_SAVE:-1}"

export KD_TOP_K="${KD_TOP_K:-16}"
export KD_LOSS="${KD_LOSS:-reverse_kl_topk}"
export KD_WEIGHT="${KD_WEIGHT:-1.0}"
export EMA_TEACHER_WEIGHT="${EMA_TEACHER_WEIGHT:-0.0}"
export TRAIN_BRANCH_MODE="${TRAIN_BRANCH_MODE:-unistream}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NUM_WORKERS="${NUM_WORKERS:-1}"
export PREFETCH="${PREFETCH:-200}"
export VALIDATE_DATA="${VALIDATE_DATA:-1}"
export VALIDATE_MAX_RECORDS="${VALIDATE_MAX_RECORDS:-2000}"
export SUMMARIZE_METRICS="${SUMMARIZE_METRICS:-1}"
export METRIC_OUTPUT_DIR="${METRIC_OUTPUT_DIR:-${REPO_ROOT}/testout/opd_metric_summary/${RUN_NAME}}"

bash run_opd_distill_llm.sh "$@"
