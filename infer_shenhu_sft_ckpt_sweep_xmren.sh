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
GPU_LIST="${GPU_LIST:-4,5,6,7}"   # used when RUN_MODE=parallel
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"        # used when RUN_MODE=parallel
ONLY_EXP="${ONLY_EXP:-}"
STREAM="${STREAM:-0}"
SPEED="${SPEED:-1.0}"
IS_SFT="${IS_SFT:-1}"              # 1 => infer_seed.py --is_sft（默认 SFT；zero-shot 请设 IS_SFT=0）
SFT_SPK_ID="${SFT_SPK_ID:-中文女}" # 需在模型 spk2info 中存在；缺则用 --auto_fallback_registered_spk 从 meta 首条注册
AUTO_FALLBACK_REGISTERED_SPK="${AUTO_FALLBACK_REGISTERED_SPK:-1}"  # 1 => --auto_fallback_registered_spk（meta 注册同名 spk）
TOP_N="${TOP_N:-5}"                # infer first N epochs (ascending); ignored when SINGLE_CKPT is set
# 若设为非空路径，则只推理该 checkpoint（不再扫描 epoch_*_whole.pt）。
# 用 SINGLE_CKPT="" 显式置空可恢复按 TOP_N 扫描（${VAR-} 仅在未设置时用默认，空字符串仍表示“扫描”）。
# SINGLE_CKPT="${SINGLE_CKPT-/home/work_nfs23/hkxie/ckpt/huawei/cosyvoice2/sft_f03_only_1e-5/epoch_2_whole.pt}"
SINGLE_CKPT="${SINGLE_CKPT-/home/work_nfs23/hkxie/ckpt/huawei/cosyvoice2/dpo_xiaoyuzhou_shenhu_10-5_1e-6_bigbatch/epoch_5_whole.pt}"
# 若设为非空目录：本次推理的 wav 与 infer.log 直接写入该目录（不再使用 OUTPUT_BASE/.../epoch_N_whole/）。
# 仅允许本轮恰好 1 个 checkpoint，否则多 epoch 会互相覆盖；请配合 SINGLE_CKPT 或 TOP_N=1。
INFER_OUT_DIR="${INFER_OUT_DIR-}"
# *_spk 类 SFT 权重与 top500k_spk 推理习惯一致，默认给 tts 加 spk 前缀；纯文本 SFT 数据请设 IS_USE_SPK_TAG=0
IS_USE_SPK_TAG="${IS_USE_SPK_TAG:-1}"   # 1 => add --is_use_spk_tag
SPK_TAG="${SPK_TAG:-<|spk_1|>}"
# 多分片：同一 lst 按「全局有效行号 % NUM_META_SHARDS == shard_index」拆给多进程；GPU_LIST 轮询绑卡；同一物理卡可绑多个进程以拉高吞吐（显存允许时 NUM_META_SHARDS >> GPU 数）。
NUM_META_SHARDS="${NUM_META_SHARDS:-1}"
# 0 => 每波同时起满 NUM_META_SHARDS（全速）；>0 为硬性并发上限（OOM 时调小，如 16）
META_SHARD_MAX_CONCURRENT="${META_SHARD_MAX_CONCURRENT:-0}"
# 1 => 恢复旧逻辑：每波最多「GPU 个数」个进程（偏保守）
META_SHARD_WAVE_CAP_GPUS="${META_SHARD_WAVE_CAP_GPUS:-0}"
# RUN_MODE=serial 时默认只用 GPU_ID 跑分片；要多卡分片设 SHARD_USE_GPU_LIST=1 并配好 GPU_LIST
SHARD_USE_GPU_LIST="${SHARD_USE_GPU_LIST:-0}"
# 多分片时终端总 tqdm（汇总各分片 .infer_progress_shard_*）；为 0 则关闭总条
INFER_TOTAL_PROGRESS_BAR="${INFER_TOTAL_PROGRESS_BAR:-1}"
# 分片日志是否 tee 到终端（总进度条开启时默认关，避免刷屏）
INFER_SHARD_TEE_PROGRESS="${INFER_SHARD_TEE_PROGRESS:-0}"
INFER_SHARD_TTY_ONLY_SHARD="${INFER_SHARD_TTY_ONLY_SHARD-}"
INFER_SHARD_PROGRESS_FLUSH="${INFER_SHARD_PROGRESS_FLUSH:-25}"
# 1 => infer_seed.py --fp16
INFER_FP16="${INFER_FP16:-0}"
# 非空 => infer_seed.py --speech_token_jsonl（多分片共写同一文件，内含 flock）
SPEECH_TOKEN_JSONL="${SPEECH_TOKEN_JSONL-}"
# 1 => infer_seed.py --unfixed_seed（覆盖 yaml 里的固定随机种子）
UNFIXED_SEED="${UNFIXED_SEED:-1}"

CKPT_ROOT="${CKPT_ROOT:-/home/work_nfs23/hkxie/ckpt/huawei/cosyvoice2}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/home/work_nfs24/xmren/Cosyvoice2-0.5B}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/xmren}"
# infer_seed.py：每行 id|prompt_text|prompt_wav_path|tts_text；prompt_wav_path 若为相对路径，则相对于「本 lst 所在目录」拼接（请与 wav 子目录一并拷贝保持相对关系）。
# META_FILE="${META_FILE:-/home/work_nfs22/xmren/code/web/data/kefu_test/meituan+shenhu_text_token_30k_tnx30_f03.lst}"
META_FILE="${META_FILE:-/home/work_nfs23/hkxie/hw_proj/testset_midterm/cmos/simluation_dpo.lst}"
META_TAG="$(basename "${META_FILE}")"
META_TAG="${META_TAG%.*}"

MODEL_DIRS=(
  "${CKPT_ROOT}/sft_f03_only_1e-5"
)

# 若 OUTPUT_BASE 下目录属主/权限导致当前用户不可写（常见于多人共用工程树），则改用可写回退目录写 summary / skip 日志 / 默认 epoch 子目录。
SUMMARY_FALLBACK_DIR="${SUMMARY_FALLBACK_DIR-}"

_run_output_base_writable() {
  local base="$1"
  local ts="${base}/summary_runs.tsv"
  mkdir -p "${base}" 2>/dev/null || return 1
  [[ -w "${base}" ]] || return 1
  if [[ -e "${ts}" ]] && [[ ! -w "${ts}" ]]; then
    return 1
  fi
  local probe="${base}/.cosyvoice_wprobe_$$"
  if ! : >"${probe}" 2>/dev/null; then
    return 1
  fi
  rm -f "${probe}"
  return 0
}

RUN_OUTPUT_BASE="${OUTPUT_BASE}/${META_TAG}"
if ! _run_output_base_writable "${RUN_OUTPUT_BASE}"; then
  _fb="${SUMMARY_FALLBACK_DIR:-${TMPDIR:-/tmp}/cosyvoice_infer_${USER:-unknown}}/${META_TAG}"
  echo "[WARN] RUN_OUTPUT_BASE not writable for current user: ${OUTPUT_BASE}/${META_TAG}" >&2
  echo "[WARN] Using RUN_OUTPUT_BASE=${_fb} (summary_runs.tsv, skip logs, default epoch dirs). Set OUTPUT_BASE=... to your own path, or SUMMARY_FALLBACK_DIR=... ." >&2
  RUN_OUTPUT_BASE="${_fb}"
  mkdir -p "${RUN_OUTPUT_BASE}" || {
    echo "[ERROR] Could not create RUN_OUTPUT_BASE: ${RUN_OUTPUT_BASE}"
    exit 1
  }
fi
SUMMARY_TSV="${RUN_OUTPUT_BASE}/summary_runs.tsv"
LOCK_FILE="${RUN_OUTPUT_BASE}/.summary_runs.lock"

if [[ ! -f "${META_FILE}" ]]; then
  echo "[ERROR] META_FILE not found: ${META_FILE}"
  exit 1
fi
if [[ ! -d "${BASE_MODEL_DIR}" ]]; then
  echo "[ERROR] BASE_MODEL_DIR not found: ${BASE_MODEL_DIR}"
  exit 1
fi

if [[ "${IS_SFT}" == "1" && -z "${SFT_SPK_ID}" ]]; then
  echo "[ERROR] IS_SFT=1 requires SFT_SPK_ID (default is 中文女; export empty only if you disable SFT)."
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
if [[ ! "${NUM_META_SHARDS}" =~ ^[0-9]+$ ]] || [[ "${NUM_META_SHARDS}" -lt 1 ]]; then
  echo "[ERROR] NUM_META_SHARDS must be a positive integer, got: ${NUM_META_SHARDS}"
  exit 1
fi
if [[ ! "${META_SHARD_MAX_CONCURRENT}" =~ ^[0-9]+$ ]] || [[ "${META_SHARD_MAX_CONCURRENT}" -lt 0 ]]; then
  echo "[ERROR] META_SHARD_MAX_CONCURRENT must be a non-negative integer, got: ${META_SHARD_MAX_CONCURRENT}"
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

_shard_fill_gpu_array() {
  SHARD_GPU_ARRAY=()
  local g
  if [[ "${PARALLEL_EXPS}" == "0" ]] && [[ "${SHARD_USE_GPU_LIST}" != "1" ]]; then
    g="$(echo "${GPU_ID}" | tr -d '[:space:]')"
    SHARD_GPU_ARRAY=("${g:-0}")
    return
  fi
  if [[ -z "${GPU_LIST// /}" ]]; then
    g="$(echo "${GPU_ID}" | tr -d '[:space:]')"
    SHARD_GPU_ARRAY=("${g:-0}")
    return
  fi
  IFS=',' read -ra SHARD_GPU_ARRAY <<< "${GPU_LIST}"
  local i gtrim
  for i in "${!SHARD_GPU_ARRAY[@]}"; do
    gtrim="$(echo "${SHARD_GPU_ARRAY[$i]}" | tr -d '[:space:]')"
    if [[ -z "${gtrim}" ]]; then
      unset "SHARD_GPU_ARRAY[$i]"
    else
      SHARD_GPU_ARRAY[$i]="${gtrim}"
    fi
  done
  SHARD_GPU_ARRAY=("${SHARD_GPU_ARRAY[@]}")
  if [[ "${#SHARD_GPU_ARRAY[@]}" -eq 0 ]]; then
    SHARD_GPU_ARRAY=("0")
  fi
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

  if [[ -n "${SINGLE_CKPT}" ]]; then
    if [[ ! -f "${SINGLE_CKPT}" ]]; then
      echo "[WARN] [GPU ${gpu_id}] SINGLE_CKPT not found: ${SINGLE_CKPT}" | tee -a "${RUN_OUTPUT_BASE}/skip_gpu${gpu_id}.log"
      append_summary_line "${gpu_id}\t${exp_name}\t${out_dir}\tmissing_single_ckpt"
      return 0
    fi
    ckpt_paths=("${SINGLE_CKPT}")
  else
    mapfile -t ckpt_paths < <(ls -1 "${model_dir}"/epoch_*_whole.pt 2>/dev/null | sort -V | head -n "${TOP_N}" || true)
    if [[ ${#ckpt_paths[@]} -eq 0 ]]; then
      echo "[WARN] [GPU ${gpu_id}] no epoch_*_whole.pt in ${model_dir}" | tee -a "${RUN_OUTPUT_BASE}/skip_gpu${gpu_id}.log"
      append_summary_line "${gpu_id}\t${exp_name}\t${out_dir}\tmissing_whole_checkpoint"
      return 0
    fi
  fi

  if [[ -n "${INFER_OUT_DIR}" ]]; then
    if [[ ${#ckpt_paths[@]} -gt 1 ]]; then
      echo "[ERROR] [GPU ${gpu_id}] INFER_OUT_DIR is set but found ${#ckpt_paths[@]} checkpoints; outputs would overwrite. Use SINGLE_CKPT or TOP_N=1, or unset INFER_OUT_DIR." \
        | tee -a "${RUN_OUTPUT_BASE}/skip_gpu${gpu_id}.log"
      append_summary_line "${gpu_id}\t${exp_name}\t-\tinfer_out_dir_multi_ckpt"
      return 1
    fi
    mkdir -p "${INFER_OUT_DIR}"
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
  if [[ "${INFER_FP16}" == "1" ]]; then
    extra_args+=(--fp16)
  fi
  if [[ -n "${SPEECH_TOKEN_JSONL}" ]]; then
    extra_args+=(--speech_token_jsonl "${SPEECH_TOKEN_JSONL}")
  fi
  if [[ "${UNFIXED_SEED}" == "1" ]]; then
    extra_args+=(--unfixed_seed)
  fi

  status="ok"
  for ckpt_path in "${ckpt_paths[@]}"; do
    ckpt_name="$(basename "${ckpt_path}")"
    epoch_num="$(echo "${ckpt_name}" | sed -E 's/^epoch_([0-9]+)_whole\.pt$/\1/')"
    if [[ -n "${INFER_OUT_DIR}" ]]; then
      epoch_out_dir="${INFER_OUT_DIR}"
    else
      epoch_out_dir="${out_dir}/epoch_${epoch_num}_whole"
    fi
    mkdir -p "${epoch_out_dir}"

    rc=0
    if [[ "${NUM_META_SHARDS}" -le 1 ]]; then
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
    else
      _shard_fill_gpu_array
      local gcount="${#SHARD_GPU_ARRAY[@]}"
      local maxc="${META_SHARD_MAX_CONCURRENT}"
      if [[ "${maxc}" -le 0 ]]; then
        if [[ "${META_SHARD_WAVE_CAP_GPUS}" == "1" ]]; then
          maxc="${NUM_META_SHARDS}"
          if [[ "${maxc}" -gt "${gcount}" ]]; then
            maxc="${gcount}"
          fi
        else
          maxc="${NUM_META_SHARDS}"
        fi
      fi
      if [[ "${maxc}" -gt "${NUM_META_SHARDS}" ]]; then
        maxc="${NUM_META_SHARDS}"
      fi
      if [[ "${maxc}" -lt 1 ]]; then
        maxc=1
      fi
      echo "[$(date +%H:%M:%S)] [GPU ${gpu_id}] exp=${exp_name} meta_shards=${NUM_META_SHARDS} max_concurrent=${maxc} shard_gpus=${SHARD_GPU_ARRAY[*]}"
      rm -f "${epoch_out_dir}"/.infer_progress_shard_*
      local META_VALID_TOTAL=0
      local prog_pid=""
      if [[ "${INFER_TOTAL_PROGRESS_BAR}" == "1" ]]; then
        META_VALID_TOTAL="$(cd "${REPO_ROOT}" && python infer_seed.py --meta_file "${META_FILE}" --count_meta_only)"
        echo "[$(date +%H:%M:%S)] [GPU ${gpu_id}] valid meta samples (total): ${META_VALID_TOTAL}"
        (
          cd "${REPO_ROOT}" && python infer_total_progress_bar.py \
            --outdir "${epoch_out_dir}" \
            --total "${META_VALID_TOTAL}" \
            --num_shards "${NUM_META_SHARDS}" \
            --interval 0.8
        ) &
        prog_pid=$!
      fi
      set +e
      local wave_start wave_end s sg
      for ((wave_start = 0; wave_start < NUM_META_SHARDS; wave_start += maxc)); do
        wave_end=$((wave_start + maxc))
        if [[ "${wave_end}" -gt "${NUM_META_SHARDS}" ]]; then
          wave_end="${NUM_META_SHARDS}"
        fi
        local wave_pids=()
        for ((s = wave_start; s < wave_end; s++)); do
          local _prog_extra=()
          sg="${SHARD_GPU_ARRAY[$((s % gcount))]}"
          echo "[$(date +%H:%M:%S)] [GPU ${gpu_id}] shard ${s}/${NUM_META_SHARDS} -> CUDA_VISIBLE_DEVICES=${sg}"
          _slog="${epoch_out_dir}/infer_shard_${s}.log"
          if [[ "${INFER_TOTAL_PROGRESS_BAR}" == "1" ]]; then
            _prog_extra+=(--shard_progress_file "${epoch_out_dir}/.infer_progress_shard_${s}")
            _prog_extra+=(--shard_progress_flush_interval "${INFER_SHARD_PROGRESS_FLUSH}")
          fi
          if [[ "${INFER_SHARD_TEE_PROGRESS}" == "1" ]]; then
            if [[ -n "${INFER_SHARD_TTY_ONLY_SHARD}" ]] && [[ "${s}" -ne "${INFER_SHARD_TTY_ONLY_SHARD}" ]]; then
              (
                export CUDA_VISIBLE_DEVICES="${sg}"
                export PYTHONUNBUFFERED=1
                python infer_seed.py \
                  --meta_file "${META_FILE}" \
                  --model_dir "${model_dir}" \
                  --base_model_dir "${BASE_MODEL_DIR}" \
                  --checkpoint_pt "${ckpt_path}" \
                  --output_dir "${epoch_out_dir}" \
                  --speed "${SPEED}" \
                  --shard_index "${s}" \
                  --num_shards "${NUM_META_SHARDS}" \
                  "${_prog_extra[@]}" \
                  "${extra_args[@]}"
              ) >"${_slog}" 2>&1 &
            else
              (
                export CUDA_VISIBLE_DEVICES="${sg}"
                export PYTHONUNBUFFERED=1
                python infer_seed.py \
                  --meta_file "${META_FILE}" \
                  --model_dir "${model_dir}" \
                  --base_model_dir "${BASE_MODEL_DIR}" \
                  --checkpoint_pt "${ckpt_path}" \
                  --output_dir "${epoch_out_dir}" \
                  --speed "${SPEED}" \
                  --shard_index "${s}" \
                  --num_shards "${NUM_META_SHARDS}" \
                  "${_prog_extra[@]}" \
                  "${extra_args[@]}"
              ) 2>&1 | tee "${_slog}" &
            fi
          else
            (
              export CUDA_VISIBLE_DEVICES="${sg}"
              export PYTHONUNBUFFERED=1
              python infer_seed.py \
                --meta_file "${META_FILE}" \
                --model_dir "${model_dir}" \
                --base_model_dir "${BASE_MODEL_DIR}" \
                --checkpoint_pt "${ckpt_path}" \
                --output_dir "${epoch_out_dir}" \
                --speed "${SPEED}" \
                --shard_index "${s}" \
                --num_shards "${NUM_META_SHARDS}" \
                "${_prog_extra[@]}" \
                "${extra_args[@]}"
            ) >"${_slog}" 2>&1 &
          fi
          wave_pids+=($!)
        done
        local p
        for p in "${wave_pids[@]}"; do
          if ! wait "${p}"; then
            rc=1
          fi
        done
      done
      if [[ -n "${prog_pid}" ]]; then
        kill "${prog_pid}" 2>/dev/null || true
        wait "${prog_pid}" 2>/dev/null || true
      fi
      set -e
    fi
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

if [[ "${NUM_META_SHARDS}" -gt 1 ]]; then
  echo "[INFO] Meta sharding: NUM_META_SHARDS=${NUM_META_SHARDS} META_SHARD_MAX_CONCURRENT=${META_SHARD_MAX_CONCURRENT} (0=all-at-once) META_SHARD_WAVE_CAP_GPUS=${META_SHARD_WAVE_CAP_GPUS} SHARD_USE_GPU_LIST=${SHARD_USE_GPU_LIST} GPU_LIST=${GPU_LIST}"
  echo "[INFO] Progress: INFER_TOTAL_PROGRESS_BAR=${INFER_TOTAL_PROGRESS_BAR} INFER_SHARD_TEE_PROGRESS=${INFER_SHARD_TEE_PROGRESS} INFER_FP16=${INFER_FP16}"
fi

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
