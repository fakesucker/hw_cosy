#!/usr/bin/env bash
# 推理 wav 与 infer.log 直接写到这里（请改成你有写权限的目录）
MY_INFER_OUT="/home/work_nfs22/hkxie/data/kefu/midterm_data"
# summary_runs.tsv、skip 日志、未设 INFER_OUT_DIR 时的分层输出根目录（同样用你自己的路径）
MY_OUTPUT_BASE="/home/work_nfs22/hkxie/data/kefu/infer_runs"

# 固定只推理该 .pt（对应目录须在 MODEL_DIRS / CKPT_ROOT 下存在 yaml 等）
MY_SINGLE_CKPT="${MY_SINGLE_CKPT:-/home/work_nfs23/hkxie/ckpt/huawei/cosyvoice2/dpo_xiaoyuzhou_shenhu_10-5_1e-6_bigbatch/epoch_5_whole.pt}"
# SFT：说话人 ID 须在 checkpoint 的 spk 表里；缺省时与仓库里 top500k_spk 推理一致。
MY_SFT_SPK_ID="${MY_SFT_SPK_ID:-中文女}"
# 与 sft_*_spk 权重一致时保持 1，会给每条 tts 前加 SPK_TAG；若你的数据无 spk 前缀则设 MY_USE_SPK_TAG=0
MY_USE_SPK_TAG="${MY_USE_SPK_TAG:-1}"

# ---------- 多卡、每卡多进程、总进度条 ----------
MY_RUN_MODE="${MY_RUN_MODE:-parallel}"
MY_GPU_LIST="${MY_GPU_LIST:-0,1,2,3,4,5,6,7}"
# 每张 GPU 上同时跑的 infer_seed 进程数（总进程数 = GPU 数 × 本值；显存不够就降到 2~4）
MY_SHARDS_PER_GPU="${MY_SHARDS_PER_GPU:-10}"
IFS=',' read -ra _GPU_ARR <<< "${MY_GPU_LIST}"
_MY_NGPU="${#_GPU_ARR[@]}"
# 若已 export MY_NUM_META_SHARDS 则优先用；否则 = GPU 数 × MY_SHARDS_PER_GPU
MY_NUM_META_SHARDS="${MY_NUM_META_SHARDS:-$((_MY_NGPU * MY_SHARDS_PER_GPU))}"
# 0 = 一次起满所有分片（最快）；OOM 时改为 8、16 等硬上限
MY_META_SHARD_MAX_CONCURRENT="${MY_META_SHARD_MAX_CONCURRENT:-0}"
# 1 = 终端一条 Total wav tqdm（默认）；0 = 关掉总条
MY_INFER_TOTAL_PROGRESS="${MY_INFER_TOTAL_PROGRESS:-1}"
# 分片日志是否 tee 到终端（总进度开启时建议 0，避免刷屏）
MY_INFER_SHARD_TEE_PROGRESS="${MY_INFER_SHARD_TEE_PROGRESS:-0}"
MY_INFER_SHARD_TTY_ONLY_SHARD="${MY_INFER_SHARD_TTY_ONLY_SHARD-}"
# 总进度汇总刷新：各分片每 N 条成功写一次计数文件
MY_INFER_SHARD_PROGRESS_FLUSH="${MY_INFER_SHARD_PROGRESS_FLUSH:-25}"
# 可选 1：infer_seed --fp16（有时略快/略省显存）
MY_INFER_FP16="${MY_INFER_FP16:-0}"
#  speech token jsonl：未设置时用 ${MY_INFER_OUT}/speech_tokens.jsonl；显式设空 MY_SPEECH_TOKEN_JSONL= 则关闭
MY_SPEECH_TOKEN_JSONL="${MY_SPEECH_TOKEN_JSONL-$MY_INFER_OUT/speech_tokens.jsonl}"
# 1 = 不固定随机种子（默认）；0 = 保持 yaml 固定 seed 可复现
MY_UNFIXED_SEED="${MY_UNFIXED_SEED:-1}"

OUTPUT_BASE="${MY_OUTPUT_BASE}" \
SINGLE_CKPT="${MY_SINGLE_CKPT}" \
INFER_OUT_DIR="${MY_INFER_OUT}" \
IS_SFT=1 \
SFT_SPK_ID="${MY_SFT_SPK_ID}" \
IS_USE_SPK_TAG="${MY_USE_SPK_TAG}" \
RUN_MODE="${MY_RUN_MODE}" \
GPU_LIST="${MY_GPU_LIST}" \
NUM_META_SHARDS="${MY_NUM_META_SHARDS}" \
META_SHARD_MAX_CONCURRENT="${MY_META_SHARD_MAX_CONCURRENT}" \
INFER_TOTAL_PROGRESS_BAR="${MY_INFER_TOTAL_PROGRESS}" \
INFER_SHARD_TEE_PROGRESS="${MY_INFER_SHARD_TEE_PROGRESS}" \
INFER_SHARD_TTY_ONLY_SHARD="${MY_INFER_SHARD_TTY_ONLY_SHARD}" \
INFER_SHARD_PROGRESS_FLUSH="${MY_INFER_SHARD_PROGRESS_FLUSH}" \
INFER_FP16="${MY_INFER_FP16}" \
SPEECH_TOKEN_JSONL="${MY_SPEECH_TOKEN_JSONL}" \
UNFIXED_SEED="${MY_UNFIXED_SEED}" \
bash /home/work_nfs23/hkxie/hw_proj/CosyVoice/infer_shenhu_sft_ckpt_sweep_xmren.sh
