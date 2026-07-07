#!/usr/bin/env bash
set -euo pipefail

# Prepare and train one OPSD run per prompt row in utt_text.scp.
# Each prompt gets its own fixed-prompt dataset, run name, checkpoint folder,
# optional non-stream eval, and optional listen_compare.html.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

CONDA_ACTIVATE="${CONDA_ACTIVATE:-/home/environment3/xmren/miniconda3/bin/activate}"
CONDA_ENV="${CONDA_ENV:-/home/environment3/xmren/miniconda3/envs/cosyvoice/}"
ACTIVATE_ENV="${ACTIVATE_ENV:-1}"
if [[ "${ACTIVATE_ENV}" == "1" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" "${CONDA_ENV}"
fi

PROMPT_SCP="${PROMPT_SCP:-/home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/prompt_wav/utt_text.scp}"
DIALOGUE_LST="${DIALOGUE_LST:-/home/work_nfs23/hkxie/hw_proj/testset_midterm/cmos/dialogue.lst}"
EXTRA_JSONL="${EXTRA_JSONL:-${REPO_ROOT}/data_list/shenhu/shenhu_filtered_wo_outliers98.jsonl}"
ONNX_PATH="${ONNX_PATH:-/home/work_nfs23/hkxie/code/CosyVoice/pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/home/work_nfs23/hkxie/ckpt/huawei/cosyvoice2/dpo_xiaoyuzhou_shenhu_10-5_1e-6_bigbatch/epoch_5_whole.pt}"
FLOW_CHECKPOINT="${FLOW_CHECKPOINT:-/home/work_nfs22/xmren/code/CosyVoice/examples/libritts/cosyvoice2/exp/flow/epoch_4_whole.pt}"

RUN_GROUP="${RUN_GROUP:-$(date +%Y%m%dT%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/data}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs_opsd}"
SWEEP_ROOT="${SWEEP_ROOT:-${REPO_ROOT}/testout/opsd_prompt_sweep_${RUN_GROUP}}"
EVAL_META_ROOT="${EVAL_META_ROOT:-${SWEEP_ROOT}/meta}"
SUMMARY_TSV="${SUMMARY_TSV:-${SWEEP_ROOT}/prompt_sweep_runs.tsv}"
RESET_SUMMARY="${RESET_SUMMARY:-0}"

PROMPT_START_INDEX="${PROMPT_START_INDEX:-1}"
PROMPT_LIMIT="${PROMPT_LIMIT:-0}"
PROMPT_ONLY="${PROMPT_ONLY:-}"
PREMIUM_FEMALE_SCP="${PREMIUM_FEMALE_SCP:-/home/work_nfs10/kxxia/bb_down_data/DB-TTS-193-2024-04-12/中文/女/female_utt_wavpath_text.scp}"
PREMIUM_MALE_SCP="${PREMIUM_MALE_SCP:-/home/work_nfs10/kxxia/bb_down_data/DB-TTS-193-2024-04-12/中文/男/male_utt_wavpath_text.scp}"
PREMIUM_PROMPTS_PER_GENDER="${PREMIUM_PROMPTS_PER_GENDER:-0}"
PREMIUM_PROMPT_SEED="${PREMIUM_PROMPT_SEED:-1986}"
PREMIUM_INCLUDE_BASE="${PREMIUM_INCLUDE_BASE:-1}"
PREMIUM_PROMPT_SCP="${PREMIUM_PROMPT_SCP:-}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
ONLY_PREPARE="${ONLY_PREPARE:-0}"
SWEEP_DRY_RUN="${SWEEP_DRY_RUN:-0}"

PREP_PROVIDER="${PREP_PROVIDER:-CPUExecutionProvider}"
CV_SIZE="${CV_SIZE:-200}"
MAX_EXTRA_RECORDS="${MAX_EXTRA_RECORDS:-0}"
ADD_SPEAKER_TAG="${ADD_SPEAKER_TAG:-0}"
SPEAKER_TAG="${SPEAKER_TAG:-<|spk_1|>}"

DISTILL_MODE="${DISTILL_MODE:-opsd}"
CONFIG="${CONFIG:-conf/cosyvoice2_sft_1e-6_spk.yaml}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-500}"
MAX_EPOCH="${MAX_EPOCH:-1000000}"
SAVE_PER_STEP="${SAVE_PER_STEP:-20}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
MAX_FRAMES_IN_BATCH="${MAX_FRAMES_IN_BATCH:-4000}"
TRAIN_BRANCH_MODE="${TRAIN_BRANCH_MODE:-unistream}"
JOIN_TIMEOUT="${JOIN_TIMEOUT:-1800}"
KD_TOP_K="${KD_TOP_K:-16}"
KD_LOSS="${KD_LOSS:-reverse_kl_topk}"
KD_WEIGHT="${KD_WEIGHT:-1.0}"
EMA_TEACHER_WEIGHT="${EMA_TEACHER_WEIGHT:-0.0}"
SKIP_CV_ON_STEP_SAVE="${SKIP_CV_ON_STEP_SAVE:-1}"
VALIDATE_DATA="${VALIDATE_DATA:-1}"
VALIDATE_MAX_RECORDS="${VALIDATE_MAX_RECORDS:-0}"
SUMMARIZE_METRICS="${SUMMARIZE_METRICS:-1}"
SHARD_DATA_LISTS="${SHARD_DATA_LISTS:-1}"
TRAIN_SHARDS="${TRAIN_SHARDS:-32}"
CV_SHARDS="${CV_SHARDS:-8}"
TRAIN_ENGINE="${TRAIN_ENGINE:-torch_ddp}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NUM_WORKERS="${NUM_WORKERS:-1}"
PREFETCH="${PREFETCH:-200}"
JOB_ID_BASE="${JOB_ID_BASE:-24000}"
RDZV_PORT_BASE="${RDZV_PORT_BASE:-1240}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-opd_distill_opsd_topk16_dialogue_shenhu_fixed}"

RUN_EVAL_AFTER_TRAIN="${RUN_EVAL_AFTER_TRAIN:-1}"
BASE_STEP="${BASE_STEP:-78520}"
EVAL_REL_STEPS="${EVAL_REL_STEPS:-20 100 200 300 400 480}"
INCLUDE_INIT="${INCLUDE_INIT:-1}"
MAX_EVAL_ROWS="${MAX_EVAL_ROWS:-20}"
MAX_TEXT_CHARS="${MAX_TEXT_CHARS:-0}"
EVAL_GPU_ID="${EVAL_GPU_ID:-3}"
EVAL_GPU_LIST="${EVAL_GPU_LIST:-${EVAL_GPU_ID}}"

mkdir -p "${LOG_ROOT}" "${SWEEP_ROOT}" "${EVAL_META_ROOT}"

require_file() {
  local path="$1"
  local name="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] ${name} not found: ${path}" >&2
    exit 1
  fi
}

contains_prompt() {
  local tag="$1"
  local idx="$2"
  local only="${PROMPT_ONLY//,/ }"
  [[ -z "${only}" ]] && return 0
  [[ " ${only} " == *" ${tag} "* || " ${only} " == *" ${idx} "* ]]
}

sanitize_tag() {
  local name="$1"
  name="${name%.*}"
  printf '%s' "${name}" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+//; s/_+$//'
}

build_prompt_scp() {
  if ! [[ "${PREMIUM_PROMPTS_PER_GENDER}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] PREMIUM_PROMPTS_PER_GENDER must be a non-negative integer, got ${PREMIUM_PROMPTS_PER_GENDER}" >&2
    exit 1
  fi
  if ! [[ "${PREMIUM_PROMPT_SEED}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] PREMIUM_PROMPT_SEED must be a non-negative integer, got ${PREMIUM_PROMPT_SEED}" >&2
    exit 1
  fi
  if (( PREMIUM_PROMPTS_PER_GENDER == 0 )); then
    return 0
  fi

  require_file "${PREMIUM_FEMALE_SCP}" PREMIUM_FEMALE_SCP
  require_file "${PREMIUM_MALE_SCP}" PREMIUM_MALE_SCP

  local include_base_arg=()
  local include_base_tag="premium_only"
  if [[ "${PREMIUM_INCLUDE_BASE}" == "1" ]]; then
    include_base_arg=(--include-base)
    include_base_tag="with_base"
  fi

  if [[ -z "${PREMIUM_PROMPT_SCP}" ]]; then
    PREMIUM_PROMPT_SCP="${EVAL_META_ROOT}/prompt_samples/${include_base_tag}_female${PREMIUM_PROMPTS_PER_GENDER}_male${PREMIUM_PROMPTS_PER_GENDER}_seed${PREMIUM_PROMPT_SEED}.scp"
  fi
  local manifest="${PREMIUM_PROMPT_SCP%.scp}.manifest.tsv"

  echo "[prompt-sweep] sample premium prompts -> ${PREMIUM_PROMPT_SCP}"
  python3 "${SCRIPT_DIR}/tools/sample_premium_prompts.py" \
    --base-prompt-scp "${PROMPT_SCP}" \
    --female-scp "${PREMIUM_FEMALE_SCP}" \
    --male-scp "${PREMIUM_MALE_SCP}" \
    --per-gender "${PREMIUM_PROMPTS_PER_GENDER}" \
    --seed "${PREMIUM_PROMPT_SEED}" \
    --output "${PREMIUM_PROMPT_SCP}" \
    --manifest "${manifest}" \
    "${include_base_arg[@]}"

  PROMPT_SCP="${PREMIUM_PROMPT_SCP}"
  echo "[prompt-sweep] effective PROMPT_SCP=${PROMPT_SCP}"
  echo "[prompt-sweep] premium manifest=${manifest}"
}

make_eval_meta() {
  local prompt_tag="$1"
  local prompt_wav="$2"
  local prompt_text="$3"
  local out_dir="${EVAL_META_ROOT}/${prompt_tag}"
  local meta_file="${out_dir}/dialogue_${prompt_tag}_prompt.lst"
  mkdir -p "${out_dir}"
  awk -F'|' -v OFS='|' -v prompt_wav="${prompt_wav}" -v prompt_text="${prompt_text}" '
    NF >= 4 {
      caption = "";
      if (NF >= 5) {
        caption = $5;
      }
      print $1, prompt_text, prompt_wav, $4, caption;
    }
  ' "${DIALOGUE_LST}" > "${meta_file}"
  printf '%s\n' "${meta_file}"
}

prepare_data() {
  local prompt_tag="$1"
  local prompt_wav="$2"
  local prompt_text="$3"
  local data_dir="$4"
  local train_list="${data_dir}/train.data.list"
  local cv_list="${data_dir}/cv.data.list"

  if [[ "${FORCE_PREPARE}" != "1" && -s "${train_list}" && -s "${cv_list}" ]]; then
    echo "[prompt:${prompt_tag}] reuse data: ${data_dir}"
    return 0
  fi

  local cmd=(
    python3 prepare_opsd_mixed_text_data.py
    --dialogue-lst "${DIALOGUE_LST}"
    --extra-jsonl "${EXTRA_JSONL}"
    --prompt-scp "${PROMPT_SCP}"
    --onnx-path "${ONNX_PATH}"
    --output-dir "${data_dir}"
    --provider "${PREP_PROVIDER}"
    --cv-size "${CV_SIZE}"
    --max-extra-records "${MAX_EXTRA_RECORDS}"
    --fixed-prompt-wav "${prompt_wav}"
    --fixed-prompt-text "${prompt_text}"
  )
  if [[ "${ADD_SPEAKER_TAG}" == "1" ]]; then
    cmd+=(--add-speaker-tag --speaker-tag "${SPEAKER_TAG}")
  fi

  echo "[prompt:${prompt_tag}] prepare data -> ${data_dir}"
  if [[ "${SWEEP_DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN]'; printf ' %q' "${cmd[@]}"; printf '\n'
  else
    "${cmd[@]}"
  fi
}

reshard_data_lists() {
  local prompt_tag="$1"
  local data_dir="$2"
  local train_list="${data_dir}/train.data.list"
  local cv_list="${data_dir}/cv.data.list"
  local shard_dir="${data_dir}/shards_${TRAIN_SHARDS}"

  if [[ "${SHARD_DATA_LISTS}" != "1" ]]; then
    return 0
  fi
  if [[ "${SWEEP_DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] shard ${data_dir}: train_shards=${TRAIN_SHARDS} cv_shards=${CV_SHARDS}"
    return 0
  fi
  require_file "${train_list}" "train.data.list for ${prompt_tag}"
  require_file "${cv_list}" "cv.data.list for ${prompt_tag}"

  local train_entries
  local cv_entries
  train_entries="$(wc -l < "${train_list}")"
  cv_entries="$(wc -l < "${cv_list}")"
  if (( train_entries > 1 && cv_entries > 1 )); then
    echo "[prompt:${prompt_tag}] data already sharded: train_entries=${train_entries} cv_entries=${cv_entries}"
    return 0
  fi

  local train_jsonl
  local cv_jsonl
  train_jsonl="$(head -n 1 "${train_list}")"
  cv_jsonl="$(head -n 1 "${cv_list}")"
  require_file "${train_jsonl}" "train jsonl for ${prompt_tag}"
  require_file "${cv_jsonl}" "cv jsonl for ${prompt_tag}"

  mkdir -p "${shard_dir}"
  cp -n "${train_list}" "${train_list}.single_jsonl.bak"
  cp -n "${cv_list}" "${cv_list}.single_jsonl.bak"
  rm -f "${shard_dir}"/train_*.jsonl "${shard_dir}"/cv_*.jsonl
  split -n "r/${TRAIN_SHARDS}" -d -a 3 --additional-suffix=.jsonl "${train_jsonl}" "${shard_dir}/train_"
  split -n "r/${CV_SHARDS}" -d -a 3 --additional-suffix=.jsonl "${cv_jsonl}" "${shard_dir}/cv_"
  find "${shard_dir}" -maxdepth 1 -type f -name 'train_*.jsonl' | sort > "${train_list}"
  find "${shard_dir}" -maxdepth 1 -type f -name 'cv_*.jsonl' | sort > "${cv_list}"
  echo "[prompt:${prompt_tag}] sharded data: train_entries=$(wc -l < "${train_list}") cv_entries=$(wc -l < "${cv_list}") dir=${shard_dir}"
}

run_train() {
  local idx="$1"
  local prompt_tag="$2"
  local data_dir="$3"
  local run_name="$4"
  local train_log="${LOG_ROOT}/${run_name}.log"
  local model_dir="${SCRIPT_DIR}/exp/cosyvoice2/${run_name}/${TRAIN_ENGINE}"
  local cuda_tag="${CUDA_VISIBLE_DEVICES//,/ }"
  cuda_tag="${cuda_tag// /_}"
  local job_id=$((JOB_ID_BASE + idx))
  local rdzv_port=$((RDZV_PORT_BASE + idx))

  echo "[prompt:${prompt_tag}] train run_name=${run_name}"
  if [[ "${SWEEP_DRY_RUN}" == "1" ]]; then
    cat <<EOF
[DRY_RUN] env DISTILL_MODE=${DISTILL_MODE} INIT_CHECKPOINT=${INIT_CHECKPOINT} TRAIN_DATA=${data_dir}/train.data.list CV_DATA=${data_dir}/cv.data.list RUN_NAME=${run_name} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS} MAX_EPOCH=${MAX_EPOCH} JOIN_TIMEOUT=${JOIN_TIMEOUT} bash run_opd_distill_llm.sh
EOF
    return 0
  fi

  DISTILL_MODE="${DISTILL_MODE}" \
  CONFIG="${CONFIG}" \
  INIT_CHECKPOINT="${INIT_CHECKPOINT}" \
  TRAIN_DATA="${data_dir}/train.data.list" \
  CV_DATA="${data_dir}/cv.data.list" \
  RUN_NAME="${run_name}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  JOB_ID="${job_id}" \
  RDZV_ENDPOINT="localhost:${rdzv_port}" \
  MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
  MAX_EPOCH="${MAX_EPOCH}" \
  SAVE_PER_STEP="${SAVE_PER_STEP}" \
  LOG_INTERVAL="${LOG_INTERVAL}" \
  MAX_FRAMES_IN_BATCH="${MAX_FRAMES_IN_BATCH}" \
  TRAIN_BRANCH_MODE="${TRAIN_BRANCH_MODE}" \
  JOIN_TIMEOUT="${JOIN_TIMEOUT}" \
  KD_TOP_K="${KD_TOP_K}" \
  KD_LOSS="${KD_LOSS}" \
  KD_WEIGHT="${KD_WEIGHT}" \
  EMA_TEACHER_WEIGHT="${EMA_TEACHER_WEIGHT}" \
  SKIP_CV_ON_STEP_SAVE="${SKIP_CV_ON_STEP_SAVE}" \
  VALIDATE_DATA="${VALIDATE_DATA}" \
  VALIDATE_MAX_RECORDS="${VALIDATE_MAX_RECORDS}" \
  SUMMARIZE_METRICS="${SUMMARIZE_METRICS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  PREFETCH="${PREFETCH}" \
  bash run_opd_distill_llm.sh 2>&1 | tee "${train_log}"

  if [[ ! -d "${model_dir}" ]]; then
    echo "[ERROR] training finished but MODEL_DIR missing: ${model_dir}" >&2
    exit 1
  fi
}

refresh_eval_compare() {
  local prompt_tag="$1"
  local prompt_wav="$2"
  local eval_meta="$3"
  local eval_output_base="$4"
  local compare_dir="${eval_output_base}/listen_compare"
  local run_root
  run_root="$(find "${eval_output_base}" -mindepth 1 -maxdepth 1 -type d ! -name listen_compare -printf '%T@ %p\n' | sort -nr | awk 'NR==1{print $2}')"
  if [[ -z "${run_root}" ]]; then
    echo "[WARN] no eval run root found under ${eval_output_base}" >&2
    return 0
  fi

  local meta_tag
  meta_tag="$(basename "${eval_meta}")"
  meta_tag="${meta_tag%.*}"
  if [[ "${MAX_EVAL_ROWS}" -gt 0 ]]; then
    meta_tag="${meta_tag}.first${MAX_EVAL_ROWS}"
  fi

  mkdir -p "${compare_dir}"
  if [[ "${INCLUDE_INIT}" == "1" ]]; then
    ln -sfn "${run_root}/infer/${meta_tag}/opsd_steps/epoch_0_whole" "${compare_dir}/baseline_init"
  fi
  for rel in ${EVAL_REL_STEPS}; do
    ln -sfn "${run_root}/infer/${meta_tag}/opsd_steps/epoch_${rel}_whole" "${compare_dir}/opsd_step${rel}"
  done

  python3 "${SCRIPT_DIR}/tools/refresh_opsd_listen_compare.py" \
    --compare-dir "${compare_dir}" \
    --meta-file "${eval_meta}" \
    --max-rows "${MAX_EVAL_ROWS}" \
    --prompt-note "Prompt: $(basename "${prompt_wav}"); mode: STREAM=0, BISTREAM_FIXED_RATIO=0; flow: $(basename "${FLOW_CHECKPOINT}"); eval rows: first ${MAX_EVAL_ROWS}."
}

run_eval() {
  local prompt_tag="$1"
  local prompt_wav="$2"
  local run_name="$3"
  local eval_meta="$4"
  local model_dir="${SCRIPT_DIR}/exp/cosyvoice2/${run_name}/${TRAIN_ENGINE}"
  local eval_output_base="${SWEEP_ROOT}/${prompt_tag}/eval"
  local eval_log="${SWEEP_ROOT}/${prompt_tag}/eval.log"
  mkdir -p "${SWEEP_ROOT}/${prompt_tag}"

  echo "[prompt:${prompt_tag}] eval -> ${eval_output_base}"
  if [[ "${SWEEP_DRY_RUN}" == "1" ]]; then
    cat <<EOF
[DRY_RUN] MODEL_DIR=${model_dir} OUTPUT_BASE=${eval_output_base} META_FILE=${eval_meta} BASE_STEP=${BASE_STEP} REL_STEPS="${EVAL_REL_STEPS}" bash run_opsd_step_ckpt_eval.sh
EOF
    return 0
  fi

  MODEL_DIR="${model_dir}" \
  OUTPUT_BASE="${eval_output_base}" \
  META_FILE="${eval_meta}" \
  BASE_STEP="${BASE_STEP}" \
  REL_STEPS="${EVAL_REL_STEPS}" \
  INCLUDE_INIT="${INCLUDE_INIT}" \
  MAX_EVAL_ROWS="${MAX_EVAL_ROWS}" \
  MAX_TEXT_CHARS="${MAX_TEXT_CHARS}" \
  FLOW_CHECKPOINT="${FLOW_CHECKPOINT}" \
  STREAM=0 \
  BISTREAM_FIXED_RATIO=0 \
  GPU_ID="${EVAL_GPU_ID}" \
  GPU_LIST="${EVAL_GPU_LIST}" \
  RUN_MODE=serial \
  EPOCH_RUN_MODE=serial \
  EPOCH_MAX_JOBS=1 \
  PROCS_PER_GPU=1 \
  DRY_RUN=0 \
  bash run_opsd_step_ckpt_eval.sh 2>&1 | tee "${eval_log}"

  refresh_eval_compare "${prompt_tag}" "${prompt_wav}" "${eval_meta}" "${eval_output_base}"
}

main() {
  require_file "${PROMPT_SCP}" PROMPT_SCP
  build_prompt_scp
  require_file "${PROMPT_SCP}" PROMPT_SCP
  require_file "${DIALOGUE_LST}" DIALOGUE_LST
  require_file "${EXTRA_JSONL}" EXTRA_JSONL
  require_file "${ONNX_PATH}" ONNX_PATH
  require_file "${INIT_CHECKPOINT}" INIT_CHECKPOINT
  require_file "${FLOW_CHECKPOINT}" FLOW_CHECKPOINT

  if [[ "${RESET_SUMMARY}" == "1" || ! -s "${SUMMARY_TSV}" ]]; then
    {
      echo -e "idx\tprompt_tag\tprompt_wav\tdata_dir\trun_name\tmodel_dir\ttrain_log\teval_meta\teval_output\tcompare_html\tstatus"
    } > "${SUMMARY_TSV}"
  fi

  local seen=0
  local selected=0
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" ]] && continue
    [[ "${line}" =~ ^# ]] && continue
    seen=$((seen + 1))
    if (( seen < PROMPT_START_INDEX )); then
      continue
    fi
    local prompt_wav="${line%%|*}"
    local prompt_text="${line#*|}"
    if [[ "${prompt_wav}" == "${line}" ]]; then
      echo "[ERROR] ${PROMPT_SCP}:${seen} expected wav|text" >&2
      exit 1
    fi
    require_file "${prompt_wav}" "prompt_wav row ${seen}"
    local prompt_tag
    prompt_tag="$(sanitize_tag "$(basename "${prompt_wav}")")"
    if ! contains_prompt "${prompt_tag}" "${seen}"; then
      continue
    fi
    if (( PROMPT_LIMIT > 0 && selected >= PROMPT_LIMIT )); then
      break
    fi
    selected=$((selected + 1))

    local data_dir="${DATA_ROOT}/opsd_dialogue_shenhu_fixed_${prompt_tag}"
    local cuda_tag="${CUDA_VISIBLE_DEVICES//,/ }"
    cuda_tag="${cuda_tag// /_}"
    local run_name="${RUN_NAME_PREFIX}_${prompt_tag}_dpo5_nonstream_mf${MAX_FRAMES_IN_BATCH}_cuda${cuda_tag}_${RUN_GROUP}"
    local model_dir="${SCRIPT_DIR}/exp/cosyvoice2/${run_name}/${TRAIN_ENGINE}"
    local train_log="${LOG_ROOT}/${run_name}.log"
    local eval_meta
    eval_meta="$(make_eval_meta "${prompt_tag}" "${prompt_wav}" "${prompt_text}")"
    local eval_output="${SWEEP_ROOT}/${prompt_tag}/eval"
    local compare_html="${eval_output}/listen_compare/listen_compare.html"

    echo "============================================================"
    echo "[prompt:${prompt_tag}] idx=${seen}/${selected}"
    echo "[prompt:${prompt_tag}] wav=${prompt_wav}"
    echo "[prompt:${prompt_tag}] data=${data_dir}"
    echo "[prompt:${prompt_tag}] run=${run_name}"
    echo "============================================================"

    prepare_data "${prompt_tag}" "${prompt_wav}" "${prompt_text}" "${data_dir}"
    reshard_data_lists "${prompt_tag}" "${data_dir}"
    if [[ "${ONLY_PREPARE}" != "1" ]]; then
      run_train "${seen}" "${prompt_tag}" "${data_dir}" "${run_name}"
      if [[ "${RUN_EVAL_AFTER_TRAIN}" == "1" ]]; then
        run_eval "${prompt_tag}" "${prompt_wav}" "${run_name}" "${eval_meta}"
      fi
    fi

    echo -e "${seen}\t${prompt_tag}\t${prompt_wav}\t${data_dir}\t${run_name}\t${model_dir}\t${train_log}\t${eval_meta}\t${eval_output}\t${compare_html}\tok" >> "${SUMMARY_TSV}"
  done < "${PROMPT_SCP}"

  if (( selected == 0 )); then
    echo "[ERROR] no prompt rows selected from ${PROMPT_SCP}" >&2
    exit 1
  fi
  echo "[prompt-sweep] selected_prompts=${selected}"
  echo "[prompt-sweep] summary=${SUMMARY_TSV}"
}

main "$@"
