#!/usr/bin/env bash
#
# infer_shenhu_sft_ckpt_sweep_init_bistream_4gpu_5_10_stream_up.sh
# ---------------------------------------------------------------
# 扫 run_all_sft_init_ckpt_seq_4gpu_bigbatch_stream_5_10.sh 产出的 7 组 SFT（bistream 5:10）。
# 每组取最近 TOP_N 个 epoch_*_whole.pt（默认 TOP_N=5），流式推理 + 客服 TN。
# 默认 BISTREAM_FIXED_RATIO=1 + BISTREAM_SPEECH_CHUNK_TOKENS=10：LLM inference_bistream [5:10]。
# STREAM=1 仅控制 Flow 音频分包；文本侧由 --bistream_fixed_ratio 驱动。
#
# 7 个实验目录（默认 CKPT_ROOT=/home/node62_data/hkxie/ckpt/huawei/cosyvoice2）：
#   sft_xiaoyuzhou_init_dpo_ep5_bistream_4gpu_5_10_bigbatch
#   sft_xiaoyuzhou_init_dpo_ep4_bistream_4gpu_5_10_bigbatch
#   sft_xiaoyuzhou_init_f03_1e5_ep2_bistream_4gpu_5_10_bigbatch
#   sft_xiaoyuzhou_init_f03_1e5_ep3_bistream_4gpu_5_10_bigbatch
#   sft_xiaoyuzhou_init_shenhu_1e6_ep2_bistream_4gpu_5_10_bigbatch
#   sft_xiaoyuzhou_init_shenhu_1e6_ep3_bistream_4gpu_5_10_bigbatch
#   sft_xiaoyuzhou_init_base_ep0_s40k_bistream_4gpu_5_10_bigbatch
#
# 默认测试集：testset_midterm/cmos/dialogue.lst（CMOS 客服对话，与 kefu_0506 同格式）
# 推荐（8 卡并行，7 实验占 7 slot）：
#   cd /home/work_nfs23/hkxie/hw_proj/CosyVoice && \
#   RUN_MODE=parallel GPU_LIST=0,1,2,3,4,5,6,7 PROCS_PER_GPU=1 TOP_N=5 \
#   ./infer_shenhu_sft_ckpt_sweep_init_bistream_4gpu_5_10_stream_up.sh
#
# 仅跑某一实验：
#   ONLY_EXP=dpo_ep5 ./infer_shenhu_sft_ckpt_sweep_init_bistream_4gpu_5_10_stream_up.sh
#
set -euo pipefail

source /home/environment2/hkxie/anaconda3/bin/activate /home/environment2/hkxie/anaconda3/envs/cosyvoice2
REPO_ROOT="/home/work_nfs23/hkxie/hw_proj/CosyVoice"
cd "${REPO_ROOT}"
export HF_ENDPOINT=https://hf-mirror.com
export COSYVOICE_CUSTOMER_SERVICE_TN=1

# --- defaults (4gpu bistream 5:10 sweep) ---
export CKPT_ROOT="${CKPT_ROOT:-/home/node62_data/hkxie/ckpt/huawei/cosyvoice2}"
export IS_SFT="${IS_SFT:-1}"
export SFT_SPK_ID="${SFT_SPK_ID:-中文女}"
export IS_USE_SPK_TAG="${IS_USE_SPK_TAG:-1}"
export SPK_TAG="${SPK_TAG:-<|spk_1|>}"
export TOP_N="${TOP_N:-5}"

DRY_RUN="${DRY_RUN:-0}"
PARALLEL_EXPS="${PARALLEL_EXPS:-1}"
RUN_MODE="${RUN_MODE:-parallel}"
GPU_ID="${GPU_ID:-0}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
ONLY_EXP="${ONLY_EXP:-}"
STREAM="${STREAM:-1}"
BISTREAM_FIXED_RATIO="${BISTREAM_FIXED_RATIO:-1}"
BISTREAM_TEXT_CHUNK_TOKENS="${BISTREAM_TEXT_CHUNK_TOKENS:-5}"
BISTREAM_SPEECH_CHUNK_TOKENS="${BISTREAM_SPEECH_CHUNK_TOKENS:-10}"
BISTREAM_FIRST_BLOCK_TEXT_TOKENS="${BISTREAM_FIRST_BLOCK_TEXT_TOKENS:-0}"
BISTREAM_FIRST_CHUNK_TEXT_TOKENS="${BISTREAM_FIRST_CHUNK_TEXT_TOKENS:-0}"
BISTREAM_FIXED_RATIO_DEBUG="${BISTREAM_FIXED_RATIO_DEBUG:-0}"
STREAM_TEXT_INPUT="${STREAM_TEXT_INPUT:-0}"
STREAM_TEXT_MIN_TOKENS="${STREAM_TEXT_MIN_TOKENS:-5}"
STREAM_TEXT_MAX_TOKENS="${STREAM_TEXT_MAX_TOKENS:-20}"
STREAM_TEXT_FIRST_CHUNK_TOKENS="${STREAM_TEXT_FIRST_CHUNK_TOKENS:-5}"
STREAM_TEXT_FORCE_CHUNK_TOKENS="${STREAM_TEXT_FORCE_CHUNK_TOKENS:-30}"
STREAM_TEXT_DEBUG="${STREAM_TEXT_DEBUG:-0}"
SENTENCE_PSEUDO_STREAM="${SENTENCE_PSEUDO_STREAM:-0}"
SENTENCE_PSEUDO_STREAM_DEBUG="${SENTENCE_PSEUDO_STREAM_DEBUG:-0}"
SPEED="${SPEED:-1.0}"
EPOCH_RUN_MODE="${EPOCH_RUN_MODE:-parallel}"
EPOCH_MAX_JOBS="${EPOCH_MAX_JOBS:-0}"
AUTO_FALLBACK_REGISTERED_SPK="${AUTO_FALLBACK_REGISTERED_SPK:-1}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/home/work_nfs24/xmren/Cosyvoice2-0.5B}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/test_init_bistream_4gpu_5_10_stream}"
META_FILE="${META_FILE:-/home/work_nfs23/hkxie/hw_proj/testset_midterm/cmos/dialogue.lst}"
META_TAG="$(basename "${META_FILE}")"
META_TAG="${META_TAG%.*}"

# Optional: space-separated extra dirs, e.g. MODEL_DIRS_OVERRIDE="/path/exp8"
if [[ -n "${MODEL_DIRS_OVERRIDE:-}" ]]; then
  # shellcheck disable=SC2206
  MODEL_DIRS=(${MODEL_DIRS_OVERRIDE})
else
  MODEL_DIRS=(
    "${CKPT_ROOT}/sft_xiaoyuzhou_init_dpo_ep5_bistream_4gpu_5_10_bigbatch"
    "${CKPT_ROOT}/sft_xiaoyuzhou_init_dpo_ep4_bistream_4gpu_5_10_bigbatch"
    "${CKPT_ROOT}/sft_xiaoyuzhou_init_f03_1e5_ep2_bistream_4gpu_5_10_bigbatch"
    "${CKPT_ROOT}/sft_xiaoyuzhou_init_f03_1e5_ep3_bistream_4gpu_5_10_bigbatch"
    "${CKPT_ROOT}/sft_xiaoyuzhou_init_shenhu_1e6_ep2_bistream_4gpu_5_10_bigbatch"
    "${CKPT_ROOT}/sft_xiaoyuzhou_init_shenhu_1e6_ep3_bistream_4gpu_5_10_bigbatch"
    "${CKPT_ROOT}/sft_xiaoyuzhou_init_base_ep0_s40k_bistream_4gpu_5_10_bigbatch"
  )
fi

RUN_OUTPUT_BASE="${OUTPUT_BASE}/${META_TAG}"
SUMMARY_TSV="${RUN_OUTPUT_BASE}/summary_runs.tsv"
LOCK_FILE="${RUN_OUTPUT_BASE}/.summary_runs.lock"
mkdir -p "${RUN_OUTPUT_BASE}"

echo "============================================================"
echo "[sweep-4gpu-5:10] init-bistream SFT x7 | TOP_N=${TOP_N} (latest epochs)"
echo "[sweep-4gpu-5:10] CKPT_ROOT=${CKPT_ROOT}"
echo "[sweep-4gpu-5:10] RUN_MODE=${RUN_MODE} GPU_LIST=${GPU_LIST} PROCS_PER_GPU=${PROCS_PER_GPU}"
echo "[sweep-4gpu-5:10] META=${META_FILE} OUT=${RUN_OUTPUT_BASE}"
echo "[sweep-4gpu-5:10] STREAM=${STREAM} BISTREAM_FIXED_RATIO=${BISTREAM_FIXED_RATIO}"
echo "[sweep-4gpu-5:10] mix_ratio text:speech = ${BISTREAM_TEXT_CHUNK_TOKENS}:${BISTREAM_SPEECH_CHUNK_TOKENS}"
echo "[sweep-4gpu-5:10] CS_TN=${COSYVOICE_CUSTOMER_SERVICE_TN:-0} IS_SFT=${IS_SFT} SFT_SPK_ID=${SFT_SPK_ID}"
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

text_mode_count=0
[[ "${BISTREAM_FIXED_RATIO}" == "1" ]] && text_mode_count=$((text_mode_count + 1))
[[ "${STREAM_TEXT_INPUT}" == "1" ]] && text_mode_count=$((text_mode_count + 1))
[[ "${SENTENCE_PSEUDO_STREAM}" == "1" ]] && text_mode_count=$((text_mode_count + 1))
if [[ "${text_mode_count}" -gt 1 ]]; then
  echo "[ERROR] text input modes are mutually exclusive. Enable only one of:"
  echo "        BISTREAM_FIXED_RATIO / STREAM_TEXT_INPUT / SENTENCE_PSEUDO_STREAM"
  exit 1
fi
if [[ ! "${BISTREAM_FIRST_BLOCK_TEXT_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] BISTREAM_FIRST_BLOCK_TEXT_TOKENS must be a non-negative integer"
  exit 1
fi
if [[ "${BISTREAM_FIRST_BLOCK_TEXT_TOKENS}" != "0" && "${BISTREAM_FIRST_BLOCK_TEXT_TOKENS}" != "6" && "${BISTREAM_FIRST_BLOCK_TEXT_TOKENS}" != "7" && "${BISTREAM_FIRST_BLOCK_TEXT_TOKENS}" != "10" ]]; then
  echo "[ERROR] BISTREAM_FIRST_BLOCK_TEXT_TOKENS must be 0, 6, 7, or 10"
  exit 1
fi
if [[ ! "${BISTREAM_TEXT_CHUNK_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] BISTREAM_TEXT_CHUNK_TOKENS must be a non-negative integer"
  exit 1
fi
if [[ ! "${BISTREAM_SPEECH_CHUNK_TOKENS}" =~ ^[0-9]+$ ]] || [[ "${BISTREAM_SPEECH_CHUNK_TOKENS}" -lt 1 ]]; then
  echo "[ERROR] BISTREAM_SPEECH_CHUNK_TOKENS must be a positive integer (default 10 for 5:10 training)"
  exit 1
fi
if [[ ! "${BISTREAM_FIRST_CHUNK_TEXT_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] BISTREAM_FIRST_CHUNK_TEXT_TOKENS must be a non-negative integer"
  exit 1
fi

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

# Pick latest TOP_N epoch_*_whole.pt (ascending sort + tail).
list_latest_ckpts() {
  local model_dir="$1"
  local n="$2"
  ls -1 "${model_dir}"/epoch_*_whole.pt 2>/dev/null | sort -V | tail -n "${n}" || true
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
    mapfile -t ckpt_paths < <(list_latest_ckpts "${model_dir}" "${TOP_N}")
    echo "[DRY_RUN] [GPU ${gpu_id}] exp=${exp_name} ckpts=${#ckpt_paths[@]} -> ${ckpt_paths[*]:-none}"
    append_summary_line "${gpu_id}\t${exp_name}\t${out_dir}\tdry_run"
    return 0
  fi

  mapfile -t ckpt_paths < <(list_latest_ckpts "${model_dir}" "${TOP_N}")
  if [[ ${#ckpt_paths[@]} -eq 0 ]]; then
    echo "[WARN] [GPU ${gpu_id}] no epoch_*_whole.pt in ${model_dir}" | tee -a "${RUN_OUTPUT_BASE}/skip_gpu${gpu_id}.log"
    append_summary_line "${gpu_id}\t${exp_name}\t${out_dir}\tmissing_whole_checkpoint"
    return 0
  fi
  echo "[GPU ${gpu_id}] exp=${exp_name} latest ${#ckpt_paths[@]} ckpts: $(basename -a "${ckpt_paths[@]}")"

  extra_args=()
  if [[ "${STREAM}" == "1" ]]; then
    extra_args+=(--stream)
  fi
  if [[ "${BISTREAM_FIXED_RATIO}" == "1" ]]; then
    extra_args+=(--bistream_fixed_ratio)
    if [[ "${BISTREAM_TEXT_CHUNK_TOKENS}" != "0" ]]; then
      extra_args+=(--bistream_text_chunk_tokens "${BISTREAM_TEXT_CHUNK_TOKENS}")
    fi
    extra_args+=(--bistream_speech_chunk_tokens "${BISTREAM_SPEECH_CHUNK_TOKENS}")
    if [[ "${BISTREAM_FIRST_BLOCK_TEXT_TOKENS}" != "0" ]]; then
      extra_args+=(--bistream_first_block_text_tokens "${BISTREAM_FIRST_BLOCK_TEXT_TOKENS}")
    fi
    if [[ "${BISTREAM_FIRST_CHUNK_TEXT_TOKENS}" != "0" ]]; then
      extra_args+=(--bistream_first_chunk_text_tokens "${BISTREAM_FIRST_CHUNK_TEXT_TOKENS}")
    fi
    if [[ "${BISTREAM_FIXED_RATIO_DEBUG}" == "1" ]]; then
      extra_args+=(--bistream_fixed_ratio_debug)
    fi
  fi
  if [[ "${STREAM_TEXT_INPUT}" == "1" ]]; then
    extra_args+=(
      --stream_text_input
      --stream_text_min_tokens "${STREAM_TEXT_MIN_TOKENS}"
      --stream_text_max_tokens "${STREAM_TEXT_MAX_TOKENS}"
      --stream_text_first_chunk_tokens "${STREAM_TEXT_FIRST_CHUNK_TOKENS}"
      --stream_text_force_chunk_tokens "${STREAM_TEXT_FORCE_CHUNK_TOKENS}"
    )
    if [[ "${STREAM_TEXT_DEBUG}" == "1" ]]; then
      extra_args+=(--stream_text_debug)
    fi
  fi
  if [[ "${SENTENCE_PSEUDO_STREAM}" == "1" ]]; then
    extra_args+=(--sentence_pseudo_stream)
    if [[ "${SENTENCE_PSEUDO_STREAM_DEBUG}" == "1" ]]; then
      extra_args+=(--sentence_pseudo_stream_debug)
    fi
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
    for ckpt_path in "${ckpt_paths[@]}"; do
      ckpt_name="$(basename "${ckpt_path}")"
      epoch_num="$(echo "${ckpt_name}" | sed -E 's/^epoch_([0-9]+)_whole\.pt$/\1/')"
      echo "[$(date +%H:%M:%S)] [GPU ${gpu_id}] launch exp=${exp_name} epoch=${epoch_num} in background"
      (
        run_epoch_job "${gpu_id}" "${model_dir}" "${ckpt_path}" "${out_dir}" "${exp_name}"
      ) &
      ep_pids+=($!)

      if [[ "${EPOCH_MAX_JOBS}" -gt 0 ]]; then
        while [[ "${#ep_pids[@]}" -ge "${EPOCH_MAX_JOBS}" ]]; do
          pid="${ep_pids[0]}"
          if ! wait "${pid}"; then
            status="partial_fail"
          fi
          ep_pids=("${ep_pids[@]:1}")
        done
      fi
    done

    for pid in "${ep_pids[@]}"; do
      if ! wait "${pid}"; then
        status="partial_fail"
      fi
    done
  fi

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
  for model_dir in "${MODEL_DIRS[@]}"; do
    run_exp_on_gpu "${GPU_ID}" "${model_dir}"
  done
fi

echo "[$(date +%H:%M:%S)] All finished. Summary: ${SUMMARY_TSV}"
