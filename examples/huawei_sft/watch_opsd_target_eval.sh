#!/usr/bin/env bash
set -euo pipefail

# Wait for selected OPSD checkpoints, run nonstream inference, and refresh the
# listening comparison page after each successful target step.

REPO_ROOT="${REPO_ROOT:-/home/work_nfs23/hkxie/hw_proj/CosyVoice}"
HUAWEI_SFT_DIR="${HUAWEI_SFT_DIR:-${REPO_ROOT}/examples/huawei_sft}"
MODEL_DIR="${MODEL_DIR:-${HUAWEI_SFT_DIR}/exp/cosyvoice2/opd_distill_opsd_topk16_dialogue_shenhu_fixed_Achird_kefu_003_dpo5_nonstream_mf4000_cuda4_7_20260628/torch_ddp}"
TRAIN_LOG="${TRAIN_LOG:-${HUAWEI_SFT_DIR}/logs_opsd/opd_distill_opsd_topk16_dialogue_shenhu_fixed_Achird_kefu_003_dpo5_nonstream_mf4000_cuda4_7_20260628.log}"
META_FILE="${META_FILE:-${REPO_ROOT}/testout/opsd_dialogue_Achird_dpo5_nonstream_init_baseline_meta/dialogue_Achird_prompt.lst}"
COMPARE_DIR="${COMPARE_DIR:-${REPO_ROOT}/testout/opsd_dialogue_Achird_dpo5_nonstream_listen_compare_20260628T095533Z}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${REPO_ROOT}/testout/opsd_dialogue_Achird_dpo5_nonstream_auto_target_eval}"
FLOW_CHECKPOINT="${FLOW_CHECKPOINT:-/home/work_nfs22/xmren/code/CosyVoice/examples/libritts/cosyvoice2/exp/flow/epoch_4_whole.pt}"

BASE_STEP="${BASE_STEP:-78520}"
TARGET_REL_STEPS="${TARGET_REL_STEPS:-200 300 400 480}"
MAX_EVAL_ROWS="${MAX_EVAL_ROWS:-20}"
POLL_SECONDS="${POLL_SECONDS:-60}"
GPU_ID="${GPU_ID:-3}"
GPU_LIST="${GPU_LIST:-${GPU_ID}}"

REFRESH_SCRIPT="${REFRESH_SCRIPT:-${HUAWEI_SFT_DIR}/tools/refresh_opsd_listen_compare.py}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${HUAWEI_SFT_DIR}/run_opsd_step_ckpt_eval.sh}"
STATE_DIR="${STATE_DIR:-${OUTPUT_PREFIX}/monitor_state}"
mkdir -p "${OUTPUT_PREFIX}" "${STATE_DIR}"

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

wav_count() {
  local wav_dir="$1"
  if [[ -d "${wav_dir}" || -L "${wav_dir}" ]]; then
    find -L "${wav_dir}" -maxdepth 1 -type f -name '*.wav' | wc -l
  else
    echo 0
  fi
}

refresh_compare() {
  python3 "${REFRESH_SCRIPT}" \
    --compare-dir "${COMPARE_DIR}" \
    --meta-file "${META_FILE}" \
    --max-rows "${MAX_EVAL_ROWS}"
}

wait_for_ckpt() {
  local rel="$1"
  local abs_step=$((BASE_STEP + rel))
  local ckpt=""
  while true; do
    ckpt="$(ls -1 "${MODEL_DIR}"/epoch_*_step_"${abs_step}".pt 2>/dev/null | sort -V | tail -n 1 || true)"
    if [[ -n "${ckpt}" ]]; then
      printf '%s\n' "${ckpt}"
      return 0
    fi
    log "waiting rel_step=${rel} abs_step=${abs_step}"
    find "${MODEL_DIR}" -maxdepth 1 -type f -name 'epoch_*_step_*.pt' -printf '%f\n' | sort -V | tail -n 5 || true
    if [[ -f "${TRAIN_LOG}" ]]; then
      tail -1 "${TRAIN_LOG}" || true
    fi
    sleep "${POLL_SECONDS}"
  done
}

run_eval_for_step() {
  local rel="$1"
  local output_base="${OUTPUT_PREFIX}/step${rel}_eval"
  local step_log="${OUTPUT_PREFIX}/step${rel}_eval.log"
  log "start eval rel_step=${rel} output_base=${output_base}"
  (
    cd "${HUAWEI_SFT_DIR}"
    MODEL_DIR="${MODEL_DIR}" \
    OUTPUT_BASE="${output_base}" \
    META_FILE="${META_FILE}" \
    BASE_STEP="${BASE_STEP}" \
    REL_STEPS="${rel}" \
    INCLUDE_INIT=0 \
    MAX_EVAL_ROWS="${MAX_EVAL_ROWS}" \
    MAX_TEXT_CHARS=0 \
    FLOW_CHECKPOINT="${FLOW_CHECKPOINT}" \
    STREAM=0 \
    BISTREAM_FIXED_RATIO=0 \
    GPU_ID="${GPU_ID}" \
    GPU_LIST="${GPU_LIST}" \
    RUN_MODE=serial \
    EPOCH_RUN_MODE=serial \
    EPOCH_MAX_JOBS=1 \
    PROCS_PER_GPU=1 \
    DRY_RUN=0 \
    bash "${EVAL_SCRIPT}"
  ) 2>&1 | tee "${step_log}"

  local run_root
  run_root="$(find "${output_base}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | awk 'NR==1{print $2}')"
  if [[ -z "${run_root}" ]]; then
    log "ERROR no run root found under ${output_base}"
    return 1
  fi

  local wav_dir="${run_root}/infer/dialogue_Achird_prompt.first20/opsd_steps/epoch_${rel}_whole"
  local count
  count="$(wav_count "${wav_dir}")"
  if [[ "${count}" -lt "${MAX_EVAL_ROWS}" ]]; then
    log "ERROR rel_step=${rel} wav_count=${count}, expected ${MAX_EVAL_ROWS}: ${wav_dir}"
    return 1
  fi

  ln -sfn "${wav_dir}" "${COMPARE_DIR}/opsd_step${rel}"
  refresh_compare
  log "done rel_step=${rel} wav_dir=${wav_dir}"
}

main() {
  log "monitor start target_steps=${TARGET_REL_STEPS}"
  log "model_dir=${MODEL_DIR}"
  log "compare_dir=${COMPARE_DIR}"
  refresh_compare

  for rel in ${TARGET_REL_STEPS}; do
    local done_marker="${STATE_DIR}/step${rel}.done"
    local failed_marker="${STATE_DIR}/step${rel}.failed"
    local existing_count
    existing_count="$(wav_count "${COMPARE_DIR}/opsd_step${rel}")"
    if [[ -f "${done_marker}" && "${existing_count}" -ge "${MAX_EVAL_ROWS}" ]]; then
      log "skip rel_step=${rel}, already done"
      continue
    fi
    rm -f "${failed_marker}"
    wait_for_ckpt "${rel}"
    if run_eval_for_step "${rel}"; then
      date '+%F %T' > "${done_marker}"
    else
      date '+%F %T' > "${failed_marker}"
      log "ERROR eval failed for rel_step=${rel}; stop monitor"
      exit 1
    fi
  done

  log "monitor complete"
}

main "$@"
