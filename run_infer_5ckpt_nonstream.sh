#!/usr/bin/env bash
#
# run_infer_5ckpt_nonstream.sh
# ---------------------------------------------------------------
# 对 5 个 checkpoint 依次做**非流式整句**推理（并行到不同 GPU）。
# 不加 --stream / --bistream_fixed_ratio，走 whole-text 整句输入路径。
#
# 5 个 checkpoint:
#   1. sft_only:  sft_shenhu_filter_wer_only_1e-5  epoch_3_whole
#   2. dpo:       dpo_xiaoyuzhou_shenhu_10-5_1e-6  epoch_5_whole
#   3. opd_turn1: premium_male10_female10           epoch_6_step_78540
#   4. opd_turn2: full_20260629_cuda4_7             epoch_6_step_78540
#   5. grpo:      grpo                              epoch_0_step_100/llm
#
set -euo pipefail

REPO_ROOT="/home/work_nfs23/hkxie/hw_proj/CosyVoice"
BASE_MODEL_DIR="/home/work_nfs24/xmren/Cosyvoice2-0.5B"
META_FILE="/home/work_nfs23/hkxie/hw_proj/testset_midterm/cmos/dialogue.lst"
OUTPUT_BASE="/home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/ckpt_comparison_5_nonstream"

source /home/environment2/hkxie/anaconda3/bin/activate /home/environment2/hkxie/anaconda3/envs/cosyvoice2
cd "${REPO_ROOT}"
export HF_ENDPOINT=https://hf-mirror.com
export COSYVOICE_CUSTOMER_SERVICE_TN=1

# name | checkpoint_pt | gpu_id
JOBS=(
  "sft_only_epoch3:/home/work_nfs23/hkxie/ckpt/huawei/cosyvoice2/sft_shenhu_filter_wer_only_1e-5_from_llm/epoch_3_whole.pt:0"
  "dpo_epoch5:/home/work_nfs23/hkxie/ckpt/huawei/cosyvoice2/dpo_xiaoyuzhou_shenhu_10-5_1e-6_bigbatch/epoch_5_whole.pt:1"
  "opd_turn1_epoch6_step78540:/home/work_nfs23/hkxie/hw_proj/CosyVoice/examples/huawei_sft/exp/cosyvoice2/opd_distill_opsd_topk16_dialogue_shenhu_fixed_turn_010_prompt_dpo5_nonstream_mf4000_cuda4_5_6_7_premium_male10_female10_20260630/torch_ddp/epoch_6_step_78540.pt:2"
  "opd_turn2_epoch6_step78540:/home/work_nfs23/hkxie/hw_proj/CosyVoice/examples/huawei_sft/exp/cosyvoice2/opd_distill_opsd_topk16_dialogue_shenhu_fixed_turn_010_prompt_dpo5_nonstream_mf4000_cuda4_5_6_7_full_20260629_cuda4_7/torch_ddp/epoch_6_step_78540.pt:3"
  "grpo_epoch0_step100:/home/node62_data/hkxie/ckpt/huawei/grpo/epoch_0_step_100/llm.pt:5"
)

echo "============================================================"
echo "[5ckpt-nonstream] $(date) Start 5-checkpoint NON-STREAM inference"
echo "[5ckpt-nonstream] Output base: ${OUTPUT_BASE}"
echo "[5ckpt-nonstream] Meta: ${META_FILE} ($(wc -l < "${META_FILE}") lines)"
echo "[5ckpt-nonstream] Mode: whole-text (no --stream, no --bistream_fixed_ratio)"
echo "============================================================"

mkdir -p "${OUTPUT_BASE}"

declare -A pids=()

run_one() {
  local name="$1"
  local ckpt="$2"
  local gpu_id="$3"
  local out_dir="${OUTPUT_BASE}/${name}"

  mkdir -p "${out_dir}"
  echo "[${name}] [$(date)] START on GPU ${gpu_id}  (non-stream, whole-text)"

  set +e
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
    python infer_seed.py \
      --meta_file "${META_FILE}" \
      --model_dir "${BASE_MODEL_DIR}" \
      --base_model_dir "${BASE_MODEL_DIR}" \
      --checkpoint_pt "${ckpt}" \
      --output_dir "${out_dir}" \
      --is_sft --sft_spk_id '中文女' \
      --is_use_spk_tag --spk_tag '<|spk_1|>' \
      --auto_fallback_registered_spk \
      --speed 1.0 \
      2>&1 | tee "${out_dir}/infer.log"
  local rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    echo "[${name}] [$(date)] DONE (ok)"
    echo "ok" > "${out_dir}/.status"
  else
    echo "[${name}] [$(date)] DONE (FAIL, exit=${rc})"
    echo "fail_${rc}" > "${out_dir}/.status"
  fi
  return $rc
}

for job in "${JOBS[@]}"; do
  IFS=':' read -r name ckpt gpu_id <<< "${job}"
  run_one "${name}" "${ckpt}" "${gpu_id}" &
  pids["${name}"]=$!
  echo "[5ckpt-nonstream] Launched ${name} (PID=${pids[${name}]}) on GPU ${gpu_id}"
done

echo ""
echo "[5ckpt-nonstream] All 5 jobs launched. Waiting for completion..."
echo ""

overall=0
for name in "${!pids[@]}"; do
  pid="${pids[${name}]}"
  if ! wait "${pid}"; then
    echo "[5ckpt-nonstream] ${name} FAILED"
    overall=1
  else
    echo "[5ckpt-nonstream] ${name} OK"
  fi
done

echo ""
echo "============================================================"
echo "[5ckpt-nonstream] $(date) ALL DONE"
echo "============================================================"
echo ""
echo "=== RESULTS ==="
for job in "${JOBS[@]}"; do
  IFS=':' read -r name ckpt gpu_id <<< "${job}"
  out_dir="${OUTPUT_BASE}/${name}"
  st="$(cat "${out_dir}/.status" 2>/dev/null || echo 'unknown')"
  wav_count="$(find "${out_dir}" -name '*.wav' 2>/dev/null | wc -l)"
  echo "  ${name}:  status=${st}  wavs=${wav_count}  dir=${out_dir}"
done
echo ""
echo "=== OUTPUT DIRECTORIES ==="
echo "  ${OUTPUT_BASE}/sft_only_epoch3/              <- sft_only"
echo "  ${OUTPUT_BASE}/dpo_epoch5/                   <- dpo"
echo "  ${OUTPUT_BASE}/opd_turn1_epoch6_step78540/   <- opd turn1"
echo "  ${OUTPUT_BASE}/opd_turn2_epoch6_step78540/   <- opd turn2"
echo "  ${OUTPUT_BASE}/grpo_epoch0_step100/          <- grpo"
echo "============================================================"
