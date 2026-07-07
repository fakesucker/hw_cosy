#!/usr/bin/env bash
# Sequential 4-GPU SFT sweep: bistream mode, text:speech mix_ratio 5:10.
# Each job uses run_sft_xiaoyuzhou_shenhu_10-5_1e-6_bigbatch_stream_4gpu_5_10.sh (CUDA 0-3, one after another).
#
# Usage:
#   bash run_all_sft_init_ckpt_seq_4gpu_bigbatch_stream_5_10.sh
#   CONTINUE_ON_FAIL=1 bash run_all_sft_init_ckpt_seq_4gpu_bigbatch_stream_5_10.sh
#   SKIP_IF_OUTPUT_EXISTS=1 bash run_all_sft_init_ckpt_seq_4gpu_bigbatch_stream_5_10.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_all_sft_init_ckpt_seq_4gpu_bigbatch_stream_5_10.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/run_sft_xiaoyuzhou_shenhu_10-5_1e-6_bigbatch_stream_4gpu_5_10.sh"
CKPT_ROOT="${CKPT_ROOT:-/home/node62_data/hkxie/ckpt/huawei/cosyvoice2}"
LOG_DIR="${SCRIPT_DIR}/logs_init_ckpt_seq_4gpu_5_10_$(date +%Y%m%d_%H%M%S)"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-0}"
SKIP_IF_OUTPUT_EXISTS="${SKIP_IF_OUTPUT_EXISTS:-0}"
export TRAIN_BRANCH_MODE="${TRAIN_BRANCH_MODE:-bistream}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "[ERROR] Missing base script: ${BASE_SCRIPT}"
  exit 1
fi

# tag | init_sft_ckpt | output_dir suffix under ${CKPT_ROOT}
JOBS=(
  "dpo_ep5|/home/work_nfs23/hkxie/ckpt/huawei/cosyvoice2/dpo_xiaoyuzhou_shenhu_10-5_1e-6_bigbatch/epoch_5_whole.pt|sft_xiaoyuzhou_init_dpo_ep5_bistream_4gpu_5_10_bigbatch"
  "dpo_ep4|/home/work_nfs23/hkxie/ckpt/huawei/cosyvoice2/dpo_xiaoyuzhou_shenhu_10-5_1e-6_bigbatch/epoch_4_whole.pt|sft_xiaoyuzhou_init_dpo_ep4_bistream_4gpu_5_10_bigbatch"
  "f03_1e5_ep2|/home/work_nfs23/hkxie/ckpt/huawei/cosyvoice2/sft_top500k_f03_10-5_1e-5_spk/epoch_2_whole.pt|sft_xiaoyuzhou_init_f03_1e5_ep2_bistream_4gpu_5_10_bigbatch"
  "f03_1e5_ep3|/home/work_nfs23/hkxie/ckpt/huawei/cosyvoice2/sft_top500k_f03_10-5_1e-5_spk/epoch_3_whole.pt|sft_xiaoyuzhou_init_f03_1e5_ep3_bistream_4gpu_5_10_bigbatch"
  "shenhu_1e6_ep2|/home/work_nfs23/hkxie/ckpt/huawei/cosyvoice2/sft_top500k_shenhu_10-5_1e-6_spk/epoch_2_whole.pt|sft_xiaoyuzhou_init_shenhu_1e6_ep2_bistream_4gpu_5_10_bigbatch"
  "shenhu_1e6_ep3|/home/work_nfs23/hkxie/ckpt/huawei/cosyvoice2/sft_top500k_shenhu_10-5_1e-6_spk/epoch_3_whole.pt|sft_xiaoyuzhou_init_shenhu_1e6_ep3_bistream_4gpu_5_10_bigbatch"
  "base_ep0_s40k|/home/work_nfs23/xmren/CosyVoice/examples/libritts/cosyvoice2/exp/shenhu_ckpt/epoch_0_step_40000.pt|sft_xiaoyuzhou_init_base_ep0_s40k_bistream_4gpu_5_10_bigbatch"
)

echo "Log dir: ${LOG_DIR}"
echo "Sequential ${#JOBS[@]} jobs (4 GPUs each, bistream mix_ratio 5:10, one job at a time)"
echo "Base script: ${BASE_SCRIPT}"
echo "CKPT_ROOT: ${CKPT_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo ""

fail=0
job_idx=0
for entry in "${JOBS[@]}"; do
  IFS='|' read -r tag init_ckpt out_name <<< "${entry}"
  output_dir="${CKPT_ROOT}/${out_name}"
  log_file="${LOG_DIR}/${tag}.log"
  job_idx=$((job_idx + 1))

  if [[ ! -f "${init_ckpt}" ]]; then
    echo "[SKIP] ${tag}: init ckpt not found: ${init_ckpt}"
    fail=1
    [[ "${CONTINUE_ON_FAIL}" == "1" ]] || exit 1
    continue
  fi

  if [[ "${SKIP_IF_OUTPUT_EXISTS}" == "1" ]]; then
    if compgen -G "${output_dir}/epoch_*_whole.pt" > /dev/null 2>&1 \
      || compgen -G "${output_dir}/epoch_*_step_*.pt" > /dev/null 2>&1; then
      echo "[SKIP] ${tag}: output already has checkpoint under ${output_dir}"
      continue
    fi
  fi

  echo "================================================================"
  echo "[${job_idx}/${#JOBS[@]}] tag=${tag}"
  echo "  init_sft_ckpt=${init_ckpt}"
  echo "  output_dir=${output_dir}"
  echo "  log=${log_file}"
  echo "  train_branch_mode=bistream, mix_ratio=5:10"
  echo "================================================================"

  export INIT_SFT_CKPT="${init_ckpt}"
  export SFT_OUTPUT_DIR="${output_dir}"
  export TRAIN_JOB_ID=$((2988 + job_idx))
  export TRAIN_RDZV_PORT=$((12459 + job_idx))

  if (
    cd "${SCRIPT_DIR}"
    bash "${BASE_SCRIPT}"
  ) >"${log_file}" 2>&1; then
    echo "[DONE] ${tag}"
  else
    echo "[FAIL] ${tag} (see ${log_file})"
    fail=1
    [[ "${CONTINUE_ON_FAIL}" == "1" ]] || exit 1
  fi
done

if [[ "${fail}" -ne 0 ]]; then
  echo "Some jobs failed or were skipped. Check logs in: ${LOG_DIR}"
  exit 1
fi

echo "All ${#JOBS[@]} init-ckpt SFT jobs completed successfully."
echo "Outputs under: ${CKPT_ROOT}/sft_xiaoyuzhou_init_*_bistream_4gpu_5_10_bigbatch"
