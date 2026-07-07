#!/usr/bin/env bash
# Original CosyVoice-style batch inference:
# - Use infer_seed.py + AutoModel(model_dir=...)
# - Prompt wav path is passed directly from meta.lst
# - No checkpoint step/whole sweep logic

set -euo pipefail

source /home/environment2/hkxie/anaconda3/bin/activate /home/environment2/hkxie/anaconda3/envs/cosyvoice2
REPO_ROOT="/home/work_nfs23/hkxie/hw_proj/CosyVoice"
cd "${REPO_ROOT}"
export HF_ENDPOINT=https://hf-mirror.com

# --- user-tunable ---
DRY_RUN="${DRY_RUN:-0}"
PARALLEL_EXPS="${PARALLEL_EXPS:-1}"
RUN_MODE="${RUN_MODE:-parallel}"   # parallel | serial
GPU_ID="${GPU_ID:-0}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"   # used when RUN_MODE=parallel
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"        # used when RUN_MODE=parallel
ONLY_EXP="${ONLY_EXP:-}"
STREAM="${STREAM:-0}"
SPEED="${SPEED:-1.0}"
IS_SFT="${IS_SFT:-0}"              # 1 => infer_seed.py --is_sft
SFT_SPK_ID="${SFT_SPK_ID:-}"       # required when IS_SFT=1
AUTO_FALLBACK_REGISTERED_SPK="${AUTO_FALLBACK_REGISTERED_SPK:-1}"  # 1 => fallback when sft spk missing
TOP_N="${TOP_N:-5}"                # infer first N epochs (ascending)
IS_USE_SPK_TAG="${IS_USE_SPK_TAG:-0}"   # 1 => add --is_use_spk_tag
SPK_TAG="${SPK_TAG:-<|spk_1|>}"

CKPT_ROOT="${CKPT_ROOT:-/home/work_nfs23/hkxie/ckpt/huawei/cosyvoice2}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/home/work_nfs24/xmren/Cosyvoice2-0.5B}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/}"
META_FILE="${META_FILE:-${REPO_ROOT}/kefu_test/kefu_0423_onlymale.lst}"
META_TAG="$(basename "${META_FILE}")"
META_TAG="${META_TAG%.*}"

MODEL_DIRS=(
  "${CKPT_ROOT}/sft_xiaoyuzhou_f03_5-5_1e-5"
  "${CKPT_ROOT}/sft_xiaoyuzhou_f03_5-5_1e-6"
  "${CKPT_ROOT}/sft_xiaoyuzhou_f03_10-5_1e-5"
  "${CKPT_ROOT}/sft_xiaoyuzhou_f03_10-5_1e-6"
  "${CKPT_ROOT}/sft_xiaoyuzhou_shenhu_5-5_1e-5"
  "${CKPT_ROOT}/sft_xiaoyuzhou_shenhu_5-5_1e-6"
  "${CKPT_ROOT}/sft_xiaoyuzhou_shenhu_10-5_1e-5"
  "${CKPT_ROOT}/sft_xiaoyuzhou_shenhu_10-5_1e-6"
)

RUN_OUTPUT_BASE="${OUTPUT_BASE}/${META_TAG}"
SUMMARY_TSV="${RUN_OUTPUT_BASE}/summary_runs.tsv"
LOCK_FILE="${RUN_OUTPUT_BASE}/.summary_runs.lock"
mkdir -p "${RUN_OUTPUT_BASE}"

if [[ ! -f "${META_FILE}" ]]; then
  echo "[ERROR] META_FILE not found: ${META_FILE}"
  exit 1
fi
if [[ ! -d "${BASE_MODEL_DIR}" ]]; then
  echo "[ERROR] BASE_MODEL_DIR not found: ${BASE_MODEL_DIR}"
  exit 1
fi

if [[ "${IS_SFT}" == "1" && -z "${SFT_SPK_ID}" ]]; then
  echo "[ERROR] IS_SFT=1 requires SFT_SPK_ID, e.g. SFT_SPK_ID='中文女'"
  exit 1
fi

# Keep backward compatibility: PARALLEL_EXPS still works, but RUN_MODE has higher priority.
if [[ "${RUN_MODE}" == "parallel" ]]; then
  PARALLEL_EXPS=1
elif [[ "${RUN_MODE}" == "serial" ]]; then
  PARALLEL_EXPS=0
else
  echo "[ERROR] RUN_MODE must be 'parallel' or 'serial', got: ${RUN_MODE}"
  exit 1
fi
if [[ ! "${PROCS_PER_GPU}" =~ ^[0-9]+$ ]] || [[ "${PROCS_PER_GPU}" -lt 1 ]]; then
  echo "[ERROR] PROCS_PER_GPU must be a positive integer, got: ${PROCS_PER_GPU}"
  exit 1
fi

append_summary_line() {
  local line="$1"
  (
    flock 200
    echo -e "${line}" >> "${SUMMARY_TSV}"
  ) 200>"${LOCK_FILE}"
}

init_summary() {
  echo -e "gpu\texp_name\toutput_dir\tstatus" > "${SUMMARY_TSV}"
}

run_exp_on_gpu() {
  local gpu_id="$1"
  local model_dir="$2"
  local exp_name
  exp_name="$(basename "${model_dir}")"

  export CUDA_VISIBLE_DEVICES="${gpu_id}"
  echo "[$(date +%H:%M:%S)] [GPU ${gpu_id}] start exp=${exp_name}"

  if [[ -n "${ONLY_EXP}" ]] && [[ "${exp_name}" != *"${ONLY_EXP}"* ]]; then
    echo "[$(date +%H:%M:%S)] [GPU ${gpu_id}] skip (ONLY_EXP): ${exp_name}"
    append_summary_line "${gpu_id}\t${exp_name}\t-\tskipped_only_exp"
    return 0
  fi

  if [[ ! -d "${model_dir}" ]]; then
    echo "[WARN] [GPU ${gpu_id}] missing dir: ${model_dir}" | tee -a "${RUN_OUTPUT_BASE}/skip_gpu${gpu_id}.log"
    append_summary_line "${gpu_id}\t${exp_name}\t-\tmissing_model_dir"
    return 0
  fi

  out_dir="${RUN_OUTPUT_BASE}/${exp_name}"
  mkdir -p "${out_dir}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    append_summary_line "${gpu_id}\t${exp_name}\t${out_dir}\tdry_run"
    return 0
  fi

  mapfile -t ckpt_paths < <(ls -1 "${model_dir}"/epoch_*_whole.pt 2>/dev/null | sort -V | head -n "${TOP_N}" || true)
  if [[ ${#ckpt_paths[@]} -eq 0 ]]; then
    echo "[WARN] [GPU ${gpu_id}] no epoch_*_whole.pt in ${model_dir}" | tee -a "${RUN_OUTPUT_BASE}/skip_gpu${gpu_id}.log"
    append_summary_line "${gpu_id}\t${exp_name}\t${out_dir}\tmissing_whole_checkpoint"
    return 0
  fi

  extra_args=()
  if [[ "${STREAM}" == "1" ]]; then
    extra_args+=(--stream)
  fi
  if [[ "${IS_SFT}" == "1" ]]; then
    extra_args+=(--is_sft --sft_spk_id "${SFT_SPK_ID}")
    if [[ "${AUTO_FALLBACK_REGISTERED_SPK}" == "1" ]]; then
      extra_args+=(--auto_fallback_registered_spk)
    fi
  fi
  if [[ "${IS_USE_SPK_TAG}" == "1" ]]; then
    extra_args+=(--is_use_spk_tag --spk_tag "${SPK_TAG}")
  fi

  status="ok"
  for ckpt_path in "${ckpt_paths[@]}"; do
    ckpt_name="$(basename "${ckpt_path}")"
    epoch_num="$(echo "${ckpt_name}" | sed -E 's/^epoch_([0-9]+)_whole\.pt$/\1/')"
    epoch_out_dir="${out_dir}/epoch_${epoch_num}_whole"
    mkdir -p "${epoch_out_dir}"

    set +e
    python infer_seed.py \
      --meta_file "${META_FILE}" \
      --model_dir "${model_dir}" \
      --base_model_dir "${BASE_MODEL_DIR}" \
      --checkpoint_pt "${ckpt_path}" \
      --output_dir "${epoch_out_dir}" \
      --speed "${SPEED}" \
      "${extra_args[@]}" \
      2>&1 | tee "${epoch_out_dir}/infer.log"
    rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then
      ep_status="ok"
    else
      ep_status="fail_exit_${rc}"
      status="partial_fail"
    fi
    append_summary_line "${gpu_id}\t${exp_name}\t${epoch_out_dir}\t${ep_status}"
  done

  # Keep one experiment-level line for quick scan.
  append_summary_line "${gpu_id}\t${exp_name}\t${out_dir}\t${status}"

  echo "[$(date +%H:%M:%S)] [GPU ${gpu_id}] done exp=${exp_name}"
}

init_summary

if [[ "${PARALLEL_EXPS}" == "1" ]]; then
  IFS=',' read -r -a gpu_array <<< "${GPU_LIST}"
  if [[ "${#gpu_array[@]}" -eq 0 ]]; then
    echo "[ERROR] GPU_LIST is empty"
    exit 1
  fi
  total_slots=$(( ${#gpu_array[@]} * PROCS_PER_GPU ))
  echo "Parallel mode: GPU_LIST=${GPU_LIST}, PROCS_PER_GPU=${PROCS_PER_GPU}, total_slots=${total_slots}"

  # Build slot list such as: 0 0 1 1 2 2 ...
  slot_gpus=()
  for g in "${gpu_array[@]}"; do
    g_trimmed="$(echo "${g}" | tr -d '[:space:]')"
    if [[ -z "${g_trimmed}" ]]; then
      continue
    fi
    for ((k=0; k<PROCS_PER_GPU; k++)); do
      slot_gpus+=("${g_trimmed}")
    done
  done
  if [[ "${#slot_gpus[@]}" -eq 0 ]]; then
    echo "[ERROR] No valid GPU slots built from GPU_LIST=${GPU_LIST}"
    exit 1
  fi

  # task queue after ONLY_EXP filtering
  task_model_dirs=()
  for model_dir in "${MODEL_DIRS[@]}"; do
    exp_name="$(basename "${model_dir}")"
    if [[ -n "${ONLY_EXP}" ]] && [[ "${exp_name}" != *"${ONLY_EXP}"* ]]; then
      continue
    fi
    task_model_dirs+=("${model_dir}")
  done

  if [[ "${#task_model_dirs[@]}" -eq 0 ]]; then
    echo "[WARN] No tasks to run after filtering."
    exit 0
  fi

  declare -a slot_pids=()
  declare -a slot_busy=()
  for ((i=0; i<${#slot_gpus[@]}; i++)); do
    slot_pids[i]=""
    slot_busy[i]=0
  done

  next_task_idx=0
  completed=0
  total_tasks=${#task_model_dirs[@]}
  ec=0

  while [[ "${completed}" -lt "${total_tasks}" ]]; do
    # Fill idle slots
    for ((s=0; s<${#slot_gpus[@]} && next_task_idx<total_tasks; s++)); do
      if [[ "${slot_busy[s]}" -eq 0 ]]; then
        gpu_id="${slot_gpus[s]}"
        model_dir="${task_model_dirs[next_task_idx]}"
        exp_name="$(basename "${model_dir}")"
        echo "[$(date +%H:%M:%S)] dispatch slot=${s} gpu=${gpu_id} exp=${exp_name}"
        ( run_exp_on_gpu "${gpu_id}" "${model_dir}" ) &
        slot_pids[s]=$!
        slot_busy[s]=1
        next_task_idx=$((next_task_idx + 1))
      fi
    done

    # Reclaim finished slots
    for ((s=0; s<${#slot_gpus[@]}; s++)); do
      if [[ "${slot_busy[s]}" -eq 1 ]]; then
        pid="${slot_pids[s]}"
        if ! kill -0 "${pid}" 2>/dev/null; then
          if ! wait "${pid}"; then
            ec=1
          fi
          slot_busy[s]=0
          slot_pids[s]=""
          completed=$((completed + 1))
        fi
      fi
    done
    sleep 1
  done

  # Just in case any unreclaimed worker remains
  for ((s=0; s<${#slot_gpus[@]}; s++)); do
    if [[ "${slot_busy[s]}" -eq 1 ]]; then
      pid="${slot_pids[s]}"
      if ! wait "${pid}"; then
        ec=1
      fi
    fi
  done

  if [[ "${ec}" -ne 0 ]]; then
    echo "[WARN] One or more parallel workers exited non-zero (check logs above)."
  fi
else
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  for i in "${!MODEL_DIRS[@]}"; do
    run_exp_on_gpu "${GPU_ID}" "${MODEL_DIRS[$i]}"
  done
fi

echo "[$(date +%H:%M:%S)] All finished. Summary: ${SUMMARY_TSV}"
