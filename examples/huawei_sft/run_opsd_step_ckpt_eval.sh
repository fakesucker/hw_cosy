#!/usr/bin/env bash
set -euo pipefail

# Stage OPSD step checkpoints as epoch_*_whole.pt and reuse the existing
# midterm inference sweep. OPSD training is unistream/full-sequence, so the
# default eval path is also whole-text, non-stream audio inference.
# Missing requested steps are reported and skipped.

REPO_ROOT="${REPO_ROOT:-/home/work_nfs23/hkxie/hw_proj/CosyVoice}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/examples/huawei_sft/exp/cosyvoice2/opd_distill_opsd_topk16/torch_ddp}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/home/work_nfs23/hkxie/code/CosyVoice/pretrained_models/CosyVoice2-0.5B}"
INFER_SCRIPT="${INFER_SCRIPT:-${REPO_ROOT}/infer_shenhu_sft_ckpt_sweep_init_bistream_0529_stream_up.sh}"
META_FILE="${META_FILE:-/home/work_nfs23/hkxie/hw_proj/testset_midterm/cmos/dialogue.lst}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/testout/opsd_step_ckpt_eval_nonstream}"
FLOW_CHECKPOINT="${FLOW_CHECKPOINT:-${COSYVOICE2_FLOW_CHECKPOINT:-/home/work_nfs22/xmren/code/CosyVoice/examples/libritts/cosyvoice2/exp/flow/epoch_4_whole.pt}}"

BASE_STEP="${BASE_STEP:-40000}"
REL_STEPS="${REL_STEPS:-20 40 60 80 100 150 200}"
INCLUDE_INIT="${INCLUDE_INIT:-1}"
MAX_EVAL_ROWS="${MAX_EVAL_ROWS:-20}"
MAX_TEXT_CHARS="${MAX_TEXT_CHARS:-0}"
DRY_RUN="${DRY_RUN:-0}"

GPU_LIST="${GPU_LIST:-0}"
RUN_MODE="${RUN_MODE:-serial}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
EPOCH_RUN_MODE="${EPOCH_RUN_MODE:-serial}"
EPOCH_MAX_JOBS="${EPOCH_MAX_JOBS:-1}"

STREAM="${STREAM:-0}"
BISTREAM_FIXED_RATIO="${BISTREAM_FIXED_RATIO:-0}"
IS_SFT="${IS_SFT:-1}"
SFT_SPK_ID="${SFT_SPK_ID:-中文女}"
IS_USE_SPK_TAG="${IS_USE_SPK_TAG:-1}"
SPK_TAG="${SPK_TAG:-<|spk_1|>}"
AUTO_FALLBACK_REGISTERED_SPK="${AUTO_FALLBACK_REGISTERED_SPK:-1}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="${OUTPUT_BASE}/${timestamp}"
STAGE_DIR="${RUN_ROOT}/staged_model_dir/opsd_steps"
SELECTED_TSV="${RUN_ROOT}/selected_step_ckpts.tsv"
mkdir -p "${STAGE_DIR}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${RUN_ROOT}/cache/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${RUN_ROOT}/cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUN_ROOT}/cache/xdg}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}/fontconfig"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "[ERROR] MODEL_DIR not found: ${MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -d "${BASE_MODEL_DIR}" ]]; then
  echo "[ERROR] BASE_MODEL_DIR not found: ${BASE_MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -f "${INFER_SCRIPT}" ]]; then
  echo "[ERROR] INFER_SCRIPT not found: ${INFER_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${META_FILE}" ]]; then
  echo "[ERROR] META_FILE not found: ${META_FILE}" >&2
  exit 1
fi
if [[ -n "${FLOW_CHECKPOINT}" ]]; then
  if [[ ! -f "${FLOW_CHECKPOINT}" ]]; then
    echo "[ERROR] FLOW_CHECKPOINT not found: ${FLOW_CHECKPOINT}" >&2
    exit 1
  fi
  export COSYVOICE2_FLOW_CHECKPOINT="${FLOW_CHECKPOINT}"
fi
if ! [[ "${BASE_STEP}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] BASE_STEP must be a non-negative integer, got ${BASE_STEP}" >&2
  exit 1
fi
if ! [[ "${MAX_EVAL_ROWS}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] MAX_EVAL_ROWS must be a non-negative integer, got ${MAX_EVAL_ROWS}" >&2
  exit 1
fi
if ! [[ "${MAX_TEXT_CHARS}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] MAX_TEXT_CHARS must be a non-negative integer, got ${MAX_TEXT_CHARS}" >&2
  exit 1
fi

echo -e "relative_step\tabsolute_step\tstatus\tcheckpoint\tstaged_as" > "${SELECTED_TSV}"
selected_count=0

stage_ckpt() {
  local rel_step="$1"
  local abs_step="$2"
  local ckpt="$3"
  local staged_name="epoch_${rel_step}_whole.pt"
  ln -sfn "${ckpt}" "${STAGE_DIR}/${staged_name}"
  echo -e "${rel_step}\t${abs_step}\tok\t${ckpt}\t${STAGE_DIR}/${staged_name}" >> "${SELECTED_TSV}"
  selected_count=$((selected_count + 1))
}

if [[ "${INCLUDE_INIT}" == "1" ]]; then
  if [[ -f "${MODEL_DIR}/init.pt" ]]; then
    stage_ckpt 0 "${BASE_STEP}" "${MODEL_DIR}/init.pt"
  else
    echo -e "0\t${BASE_STEP}\tmissing\t${MODEL_DIR}/init.pt\t" >> "${SELECTED_TSV}"
  fi
fi

for rel_step in ${REL_STEPS}; do
  if ! [[ "${rel_step}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] REL_STEPS contains non-integer value: ${rel_step}" >&2
    exit 1
  fi
  abs_step=$((BASE_STEP + rel_step))
  ckpt="$(ls -1 "${MODEL_DIR}"/epoch_*_step_"${abs_step}".pt 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -n "${ckpt}" ]]; then
    stage_ckpt "${rel_step}" "${abs_step}" "${ckpt}"
  else
    echo -e "${rel_step}\t${abs_step}\tmissing\t${MODEL_DIR}/epoch_*_step_${abs_step}.pt\t" >> "${SELECTED_TSV}"
  fi
done

if [[ "${selected_count}" -eq 0 ]]; then
  echo "[ERROR] no checkpoints staged from ${MODEL_DIR}" >&2
  cat "${SELECTED_TSV}" >&2
  exit 1
fi

EVAL_META="${META_FILE}"
if [[ "${MAX_EVAL_ROWS}" -gt 0 ]]; then
  meta_tag="$(basename "${META_FILE}")"
  meta_tag="${meta_tag%.*}"
  EVAL_META="${RUN_ROOT}/${meta_tag}.first${MAX_EVAL_ROWS}.lst"
  meta_dir="$(dirname "${META_FILE}")"
  awk -F'|' -v OFS='|' -v max="${MAX_EVAL_ROWS}" -v max_chars="${MAX_TEXT_CHARS}" -v meta_dir="${meta_dir}" '
    NF >= 4 {
      if (max_chars > 0 && length($4) > max_chars) {
        next;
      }
      wav=$3;
      if (wav !~ /^\//) {
        wav=meta_dir "/" wav;
      }
      print $1, $2, wav, $4;
      count++;
      if (count >= max) {
        exit;
      }
    }
  ' "${META_FILE}" > "${EVAL_META}"
  if [[ ! -s "${EVAL_META}" ]]; then
    echo "[ERROR] no eval rows selected from ${META_FILE}; try increasing MAX_TEXT_CHARS" >&2
    exit 1
  fi
fi

echo "[opsd-step-eval] MODEL_DIR=${MODEL_DIR}"
echo "[opsd-step-eval] STAGE_DIR=${STAGE_DIR}"
echo "[opsd-step-eval] selected_count=${selected_count}"
echo "[opsd-step-eval] selected -> ${SELECTED_TSV}"
echo "[opsd-step-eval] META=${EVAL_META}"
echo "[opsd-step-eval] MAX_EVAL_ROWS=${MAX_EVAL_ROWS} MAX_TEXT_CHARS=${MAX_TEXT_CHARS}"
echo "[opsd-step-eval] FLOW_CHECKPOINT=${FLOW_CHECKPOINT}"
cat "${SELECTED_TSV}"

MODEL_DIRS_OVERRIDE="${STAGE_DIR}" \
BASE_MODEL_DIR="${BASE_MODEL_DIR}" \
COSYVOICE2_FLOW_CHECKPOINT="${FLOW_CHECKPOINT}" \
OUTPUT_BASE="${RUN_ROOT}/infer" \
META_FILE="${EVAL_META}" \
TOP_N="${selected_count}" \
RUN_MODE="${RUN_MODE}" \
GPU_LIST="${GPU_LIST}" \
PROCS_PER_GPU="${PROCS_PER_GPU}" \
EPOCH_RUN_MODE="${EPOCH_RUN_MODE}" \
EPOCH_MAX_JOBS="${EPOCH_MAX_JOBS}" \
DRY_RUN="${DRY_RUN}" \
STREAM="${STREAM}" \
BISTREAM_FIXED_RATIO="${BISTREAM_FIXED_RATIO}" \
IS_SFT="${IS_SFT}" \
SFT_SPK_ID="${SFT_SPK_ID}" \
IS_USE_SPK_TAG="${IS_USE_SPK_TAG}" \
SPK_TAG="${SPK_TAG}" \
AUTO_FALLBACK_REGISTERED_SPK="${AUTO_FALLBACK_REGISTERED_SPK}" \
bash "${INFER_SCRIPT}"

echo "[opsd-step-eval] done: ${RUN_ROOT}"
