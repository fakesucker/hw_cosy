#!/usr/bin/env bash
#
# infer_shenhu_sft_ckpt_sweep_onlyshenhu_dpo_tn.sh
# -----------------------------------------------
# 客服 kefu 列表 + 客服预 TN（COSYVOICE_CUSTOMER_SERVICE_TN=1）+ SFT 推理扫 ckpt。
#
# 本 sweep 固定对比两个实验目录（MODEL_DIRS）：
#   1) ${CKPT_ROOT}/sft_shenhu_only_1e-5_from_llm
#   2) ${CKPT_ROOT}/dpo_xiaoyuzhou_shenhu_10-5_1e-6_bigbatch
#
# 8 卡并行（默认 GPU_LIST=0–7，RUN_MODE=parallel）：每个实验占一个 slot，2 个实验
# 同时跑在 GPU0、GPU1；其余卡空闲。若 MODEL_DIRS 增至 8 个及以下实验，可占满 8 卡。
#
# 推荐一键命令（8 卡机器、并行跑上述两个 exp）：
#   cd /path/to/CosyVoice && \
#   RUN_MODE=parallel GPU_LIST=0,1,2,3,4,5,6,7 PROCS_PER_GPU=1 \
#   ./infer_shenhu_sft_ckpt_sweep_onlyshenhu_dpo_tn.sh
#
# 其它行为：
# - infer_seed.py + AutoModel；meta 里带 prompt；按 epoch_*_whole.pt 扫 TOP_N 个 ckpt
# - 流式：STREAM=1 时 CosyVoice2/3 在 model 内固定首包 16 speech token（~0.64s@25Hz）、之后每包 self.token_hop_len（默认 25，~1s）
#
set -euo pipefail

source /home/environment2/hkxie/anaconda3/bin/activate /home/environment2/hkxie/anaconda3/envs/cosyvoice2
REPO_ROOT="/home/work_nfs23/hkxie/hw_proj/CosyVoice"
cd "${REPO_ROOT}"
export HF_ENDPOINT=https://hf-mirror.com
export COSYVOICE_CUSTOMER_SERVICE_TN=1 # 规则匹配开关 tn 生效
# SPK training dedicated inference defaults.
export CKPT_ROOT="${CKPT_ROOT:-/home/work_nfs23/hkxie/ckpt/huawei/cosyvoice2}"
export MODEL_DIRS_OVERRIDE="${MODEL_DIRS_OVERRIDE:-}"
export IS_SFT="${IS_SFT:-1}"
export SFT_SPK_ID="${SFT_SPK_ID:-中文女}"
export IS_USE_SPK_TAG="${IS_USE_SPK_TAG:-1}"
export SPK_TAG="${SPK_TAG:-<|spk_1|>}"
export TOP_N="${TOP_N:-5}"

# --- user-tunable ---
DRY_RUN="${DRY_RUN:-0}"
PARALLEL_EXPS="${PARALLEL_EXPS:-1}"
RUN_MODE="${RUN_MODE:-parallel}"   # parallel | serial
GPU_ID="${GPU_ID:-0}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"   # used when RUN_MODE=parallel
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"        # used when RUN_MODE=parallel
ONLY_EXP="${ONLY_EXP:-}"
STREAM="${STREAM:-1}"
SPEED="${SPEED:-1.0}"
EPOCH_RUN_MODE="${EPOCH_RUN_MODE:-parallel}"   # parallel | serial (inside one exp)
EPOCH_MAX_JOBS="${EPOCH_MAX_JOBS:-0}"          # 0 => no limit in parallel mode
IS_SFT="${IS_SFT:-1}"              # 1 => infer_seed.py --is_sft
SFT_SPK_ID="${SFT_SPK_ID:-}"       # required when IS_SFT=1
AUTO_FALLBACK_REGISTERED_SPK="${AUTO_FALLBACK_REGISTERED_SPK:-1}"  # 1 => fallback when sft spk missing
TOP_N="${TOP_N:-5}"                # infer first N epochs (ascending)
IS_USE_SPK_TAG="${IS_USE_SPK_TAG:-1}"   # 1 => add --is_use_spk_tag
SPK_TAG="${SPK_TAG:-<|spk_1|>}"

CKPT_ROOT="${CKPT_ROOT:-/home/work_nfs23/hkxie/ckpt/huawei/cosyvoice2}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/home/work_nfs24/xmren/Cosyvoice2-0.5B}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/test_midterm_0524}"
META_FILE="${META_FILE:-/home/work_nfs23/hkxie/hw_proj/testset_midterm/wer/meituan_ceping_cer_wer_set2.lst}" # /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/kefu_0506_onlymale.lst
META_TAG="$(basename "${META_FILE}")"
META_TAG="${META_TAG%.*}"

MODEL_DIRS=(
  "${CKPT_ROOT}/dpo_xiaoyuzhou_shenhu_10-5_1e-6"
  "${CKPT_ROOT}/dpo_xiaoyuzhou_shenhu_10-5_1e-6_bigbatch"
  "${CKPT_ROOT}/sft_top500k_shenhu_10-5_1e-5_spk"
  "${CKPT_ROOT}/sft_top500k_shenhu_10-5_1e-6_spk"
  "${CKPT_ROOT}/sft_shenhu_only_1e-5_from_llm"
)

RUN_OUTPUT_BASE="${OUTPUT_BASE}/${META_TAG}"
SUMMARY_TSV="${RUN_OUTPUT_BASE}/summary_runs.tsv"
LOCK_FILE="${RUN_OUTPUT_BASE}/.summary_runs.lock"
mkdir -p "${RUN_OUTPUT_BASE}"

echo "============================================================"
echo "[sweep] sft_shenhu_only_1e-5_from_llm  vs  dpo_xiaoyuzhou_shenhu_10-5_1e-6_bigbatch"
echo "[sweep] RUN_MODE=${RUN_MODE} GPU_LIST=${GPU_LIST} PROCS_PER_GPU=${PROCS_PER_GPU}"
echo "[sweep] META=${META_FILE} OUT=${RUN_OUTPUT_BASE} TOP_N=${TOP_N} CS_TN=${COSYVOICE_CUSTOMER_SERVICE_TN:-0} STREAM=${STREAM}"
echo "============================================================"

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
if [[ "${EPOCH_RUN_MODE}" != "parallel" && "${EPOCH_RUN_MODE}" != "serial" ]]; then
  echo "[ERROR] EPOCH_RUN_MODE must be 'parallel' or 'serial', got: ${EPOCH_RUN_MODE}"
  exit 1
fi
if [[ ! "${EPOCH_MAX_JOBS}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] EPOCH_MAX_JOBS must be a non-negative integer, got: ${EPOCH_MAX_JOBS}"
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
  run_epoch_job() {
    local gpu_id="$1"
    local model_dir="$2"
    local ckpt_path="$3"
    local out_dir="$4"
    local exp_name="$5"
    local ckpt_name epoch_num epoch_out_dir rc ep_status
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
    fi
    append_summary_line "${gpu_id}\t${exp_name}\t${epoch_out_dir}\t${ep_status}"
    return "${rc}"
  }

  if [[ "${EPOCH_RUN_MODE}" == "serial" ]]; then
    for ckpt_path in "${ckpt_paths[@]}"; do
      if ! run_epoch_job "${gpu_id}" "${model_dir}" "${ckpt_path}" "${out_dir}" "${exp_name}"; then
        status="partial_fail"
      fi
    done
  else
    declare -a ep_pids=()
    declare -a ep_ckpts=()
    for ckpt_path in "${ckpt_paths[@]}"; do
      ckpt_name="$(basename "${ckpt_path}")"
      epoch_num="$(echo "${ckpt_name}" | sed -E 's/^epoch_([0-9]+)_whole\.pt$/\1/')"
      echo "[$(date +%H:%M:%S)] [GPU ${gpu_id}] launch exp=${exp_name} epoch=${epoch_num} in background"
      (
        run_epoch_job "${gpu_id}" "${model_dir}" "${ckpt_path}" "${out_dir}" "${exp_name}"
      ) &
      ep_pids+=($!)
      ep_ckpts+=("${ckpt_path}")

      if [[ "${EPOCH_MAX_JOBS}" -gt 0 ]]; then
        while [[ "${#ep_pids[@]}" -ge "${EPOCH_MAX_JOBS}" ]]; do
          pid="${ep_pids[0]}"
          if ! wait "${pid}"; then
            status="partial_fail"
          fi
          ep_pids=("${ep_pids[@]:1}")
          ep_ckpts=("${ep_ckpts[@]:1}")
        done
      fi
    done

    for pid in "${ep_pids[@]}"; do
      if ! wait "${pid}"; then
        status="partial_fail"
      fi
    done
  fi

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
