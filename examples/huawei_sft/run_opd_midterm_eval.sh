#!/usr/bin/env bash
set -euo pipefail

# Evaluate baseline/OPD CosyVoice2 checkpoints on testset_midterm set1/set2/set3.
#
# Example:
#   OPD_MODEL_DIRS="/path/to/opd/torch_ddp" \
#   BASELINE_MODEL_DIRS="/path/to/baseline/torch_ddp" \
#   TOP_N=1 RUN_MODE=serial GPU_LIST=0 bash run_opd_midterm_eval.sh

REPO_ROOT="${REPO_ROOT:-/home/work_nfs23/hkxie/hw_proj/CosyVoice}"
MIDTERM_ROOT="${MIDTERM_ROOT:-/home/work_nfs23/hkxie/hw_proj/testset_midterm}"
INFER_SCRIPT="${INFER_SCRIPT:-${REPO_ROOT}/infer_shenhu_sft_ckpt_sweep_init_bistream_0529_stream_up.sh}"
MIDTERM_SCRIPT_DIR="${MIDTERM_SCRIPT_DIR:-${MIDTERM_ROOT}/srcipt}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/home/work_nfs23/hkxie/code/CosyVoice/pretrained_models/CosyVoice2-0.5B}"
OPD_MODEL_DIRS="${OPD_MODEL_DIRS:?Set OPD_MODEL_DIRS='/path/to/opd/torch_ddp [...]'}"
BASELINE_MODEL_DIRS="${BASELINE_MODEL_DIRS:-}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/testout/opd_midterm_eval}"
TOP_N="${TOP_N:-1}"
RUN_INFER="${RUN_INFER:-1}"
RUN_CER="${RUN_CER:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_MODE="${RUN_MODE:-serial}"
GPU_LIST="${GPU_LIST:-0}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
EPOCH_RUN_MODE="${EPOCH_RUN_MODE:-serial}"
EPOCH_MAX_JOBS="${EPOCH_MAX_JOBS:-1}"
WER_NUM_JOB="${WER_NUM_JOB:-4}"
WER_GPU_OFFSET="${WER_GPU_OFFSET:-0}"
VALIDATE_EVAL_INPUTS="${VALIDATE_EVAL_INPUTS:-1}"
EVAL_PREFLIGHT_TSV="${EVAL_PREFLIGHT_TSV:-${OUTPUT_BASE}/preflight_model_dirs.tsv}"
STAGE_EVAL_MODEL_DIRS="${STAGE_EVAL_MODEL_DIRS:-1}"
EVAL_STAGE_DIR="${EVAL_STAGE_DIR:-${OUTPUT_BASE}/staged_model_dirs/$(date -u +%Y%m%dT%H%M%SZ)}"
EVAL_STAGED_TSV="${EVAL_STAGED_TSV:-${OUTPUT_BASE}/staged_model_dirs.tsv}"
SUMMARIZE_LATENCY="${SUMMARIZE_LATENCY:-1}"
LATENCY_SUMMARY_TSV="${LATENCY_SUMMARY_TSV:-${OUTPUT_BASE}/summary_latency.tsv}"
COMPARE_EVAL="${COMPARE_EVAL:-1}"
COMPARE_SUMMARY_TSV="${COMPARE_SUMMARY_TSV:-${OUTPUT_BASE}/summary_compare.tsv}"
if [[ "${DRY_RUN}" == "1" ]]; then
  REQUIRE_CER_ROWS="${REQUIRE_CER_ROWS:-0}"
  REQUIRE_CER_OK="${REQUIRE_CER_OK:-0}"
  REQUIRE_COMPARE_ROWS="${REQUIRE_COMPARE_ROWS:-0}"
else
  REQUIRE_CER_ROWS="${REQUIRE_CER_ROWS:-1}"
  REQUIRE_CER_OK="${REQUIRE_CER_OK:-1}"
  REQUIRE_COMPARE_ROWS="${REQUIRE_COMPARE_ROWS:-1}"
fi

META_SET1="${META_SET1:-${MIDTERM_ROOT}/wer/ceping_cer_wer.lst}"
META_SET2="${META_SET2:-${MIDTERM_ROOT}/wer/ceping_cer_wer_set2.lst}"
META_SET3="${META_SET3:-${MIDTERM_ROOT}/wer/ceping_cer_wer_set3.lst}"

mkdir -p "${OUTPUT_BASE}"

is_positive_int() {
  [[ "$1" =~ ^[0-9]+$ ]] && [[ "$1" -gt 0 ]]
}

if [[ ! -f "${INFER_SCRIPT}" ]]; then
  echo "[ERROR] INFER_SCRIPT not found: ${INFER_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${MIDTERM_SCRIPT_DIR}/run_ceping_wer.sh" ]]; then
  echo "[ERROR] run_ceping_wer.sh not found under ${MIDTERM_SCRIPT_DIR}" >&2
  exit 1
fi
for meta in "${META_SET1}" "${META_SET2}" "${META_SET3}"; do
  if [[ ! -f "${meta}" ]]; then
    echo "[ERROR] meta file not found: ${meta}" >&2
    exit 1
  fi
done
if [[ ! -d "${BASE_MODEL_DIR}" ]]; then
  echo "[ERROR] BASE_MODEL_DIR not found: ${BASE_MODEL_DIR}" >&2
  exit 1
fi
if ! is_positive_int "${TOP_N}"; then
  echo "[ERROR] TOP_N must be a positive integer, got: ${TOP_N}" >&2
  exit 1
fi
if ! is_positive_int "${PROCS_PER_GPU}"; then
  echo "[ERROR] PROCS_PER_GPU must be a positive integer, got: ${PROCS_PER_GPU}" >&2
  exit 1
fi
if [[ ! "${EPOCH_MAX_JOBS}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] EPOCH_MAX_JOBS must be a non-negative integer, got: ${EPOCH_MAX_JOBS}" >&2
  exit 1
fi
if ! is_positive_int "${WER_NUM_JOB}"; then
  echo "[ERROR] WER_NUM_JOB must be a positive integer, got: ${WER_NUM_JOB}" >&2
  exit 1
fi
if [[ ! "${WER_GPU_OFFSET}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] WER_GPU_OFFSET must be a non-negative integer, got: ${WER_GPU_OFFSET}" >&2
  exit 1
fi

ALL_MODEL_DIRS_ORIGINAL="${BASELINE_MODEL_DIRS:+${BASELINE_MODEL_DIRS} }${OPD_MODEL_DIRS}"
ALL_MODEL_DIRS="${ALL_MODEL_DIRS_ORIGINAL}"
echo "[opd-eval] REPO_ROOT=${REPO_ROOT}"
echo "[opd-eval] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[opd-eval] MODEL_DIRS=${ALL_MODEL_DIRS}"
echo "[opd-eval] TOP_N=${TOP_N} RUN_INFER=${RUN_INFER} RUN_CER=${RUN_CER}"

validate_model_dirs() {
  local role="$1"
  local dirs_text="$2"
  local -a dirs
  # shellcheck disable=SC2206
  dirs=(${dirs_text})
  if [[ "${#dirs[@]}" -eq 0 ]]; then
    if [[ "${role}" == "opd" ]]; then
      echo "[ERROR] OPD_MODEL_DIRS is empty" >&2
      return 1
    fi
    return 0
  fi

  local status=0 dir exp_name ckpt_count latest_ckpts ckpt_list
  for dir in "${dirs[@]}"; do
    exp_name="$(basename "${dir}")"
    if [[ ! -d "${dir}" ]]; then
      echo -e "${role}\t${exp_name}\t${dir}\t0\tmissing_dir\t" >> "${EVAL_PREFLIGHT_TSV}"
      echo "[ERROR] ${role} model dir not found: ${dir}" >&2
      status=1
      continue
    fi
    latest_ckpts="$(ls -1 "${dir}"/epoch_*_whole.pt 2>/dev/null | sort -V | tail -n "${TOP_N}" || true)"
    ckpt_count=0
    ckpt_list=""
    if [[ -n "${latest_ckpts}" ]]; then
      ckpt_count="$(echo "${latest_ckpts}" | wc -l | tr -d ' ')"
      ckpt_list="$(echo "${latest_ckpts}" | xargs -r -n1 basename | paste -sd ',' -)"
    fi
    if [[ "${ckpt_count}" -eq 0 ]]; then
      echo -e "${role}\t${exp_name}\t${dir}\t0\tmissing_epoch_whole\t" >> "${EVAL_PREFLIGHT_TSV}"
      echo "[ERROR] no epoch_*_whole.pt found in ${role} model dir: ${dir}" >&2
      status=1
      continue
    fi
    echo -e "${role}\t${exp_name}\t${dir}\t${ckpt_count}\tok\t${ckpt_list}" >> "${EVAL_PREFLIGHT_TSV}"
  done
  return "${status}"
}

if [[ "${VALIDATE_EVAL_INPUTS}" == "1" ]]; then
  echo -e "role\texp\tmodel_dir\tselected_ckpt_count\tstatus\tselected_ckpts" > "${EVAL_PREFLIGHT_TSV}"
  validate_model_dirs "baseline" "${BASELINE_MODEL_DIRS}"
  validate_model_dirs "opd" "${OPD_MODEL_DIRS}"
  echo "[opd-eval] preflight -> ${EVAL_PREFLIGHT_TSV}"
fi

model_alias() {
  local role="$1"
  local dir="$2"
  local base parent alias
  base="$(basename "${dir}")"
  if [[ "${base}" == "torch_ddp" || "${base}" == "deepspeed" ]]; then
    parent="$(basename "$(dirname "${dir}")")"
    alias="${parent}"
  else
    alias="${base}"
  fi
  alias="$(echo "${alias}" | sed 's/[^A-Za-z0-9_.-]/_/g')"
  echo "${role}_${alias}"
}

unique_model_alias() {
  local role="$1"
  local dir="$2"
  local alias count
  alias="$(model_alias "${role}" "${dir}")"
  count="${EVAL_ALIAS_COUNTS[${alias}]:-0}"
  count=$((count + 1))
  EVAL_ALIAS_COUNTS["${alias}"]="${count}"
  if [[ "${count}" -eq 1 ]]; then
    UNIQUE_MODEL_ALIAS="${alias}"
  else
    UNIQUE_MODEL_ALIAS="${alias}_${count}"
  fi
}

stage_model_dirs() {
  local role="$1"
  local dirs_text="$2"
  local -a dirs
  # shellcheck disable=SC2206
  dirs=(${dirs_text})
  if [[ "${#dirs[@]}" -eq 0 ]]; then
    return 0
  fi

  local dir alias stage_dir latest_ckpts ckpt ckpt_count ckpt_list
  for dir in "${dirs[@]}"; do
    unique_model_alias "${role}" "${dir}"
    alias="${UNIQUE_MODEL_ALIAS}"
    stage_dir="${EVAL_STAGE_DIR}/${alias}"
    mkdir -p "${stage_dir}"
    latest_ckpts="$(ls -1 "${dir}"/epoch_*_whole.pt 2>/dev/null | sort -V | tail -n "${TOP_N}" || true)"
    if [[ -z "${latest_ckpts}" ]]; then
      echo "[ERROR] cannot stage ${role} model dir with no epoch_*_whole.pt: ${dir}" >&2
      return 1
    fi
    ckpt_count=0
    ckpt_list=""
    while IFS= read -r ckpt; do
      [[ -n "${ckpt}" ]] || continue
      ln -sfn "${ckpt}" "${stage_dir}/$(basename "${ckpt}")"
      ckpt_count=$((ckpt_count + 1))
      if [[ -z "${ckpt_list}" ]]; then
        ckpt_list="$(basename "${ckpt}")"
      else
        ckpt_list="${ckpt_list},$(basename "${ckpt}")"
      fi
    done <<< "${latest_ckpts}"
    echo -e "${role}\t${alias}\t${dir}\t${stage_dir}\t${ckpt_count}\t${ckpt_list}" >> "${EVAL_STAGED_TSV}"
    EVAL_MODEL_DIRS_ARRAY+=("${stage_dir}")
  done
}

if [[ "${STAGE_EVAL_MODEL_DIRS}" == "1" ]]; then
  EVAL_MODEL_DIRS_ARRAY=()
  declare -A EVAL_ALIAS_COUNTS=()
  echo -e "role\teval_exp\toriginal_model_dir\tstaged_model_dir\tselected_ckpt_count\tselected_ckpts" > "${EVAL_STAGED_TSV}"
  stage_model_dirs "baseline" "${BASELINE_MODEL_DIRS}"
  stage_model_dirs "opd" "${OPD_MODEL_DIRS}"
  ALL_MODEL_DIRS="${EVAL_MODEL_DIRS_ARRAY[*]}"
  echo "[opd-eval] staged model dirs -> ${EVAL_STAGED_TSV}"
  echo "[opd-eval] staged MODEL_DIRS=${ALL_MODEL_DIRS}"
fi

run_infer_for_meta() {
  local meta_file="$1"
  local meta_tag
  meta_tag="$(basename "${meta_file}")"
  meta_tag="${meta_tag%.*}"
  echo "========== Infer ${meta_tag} =========="
  MODEL_DIRS_OVERRIDE="${ALL_MODEL_DIRS}" \
  BASE_MODEL_DIR="${BASE_MODEL_DIR}" \
  OUTPUT_BASE="${OUTPUT_BASE}/infer" \
  META_FILE="${meta_file}" \
  TOP_N="${TOP_N}" \
  RUN_MODE="${RUN_MODE}" \
  GPU_LIST="${GPU_LIST}" \
  PROCS_PER_GPU="${PROCS_PER_GPU}" \
  EPOCH_RUN_MODE="${EPOCH_RUN_MODE}" \
  EPOCH_MAX_JOBS="${EPOCH_MAX_JOBS}" \
  DRY_RUN="${DRY_RUN}" \
  bash "${INFER_SCRIPT}"
}

run_cer_for_meta() {
  local set_name="$1"
  local meta_file="$2"
  local summary_all="$3"
  local meta_tag infer_meta_root
  meta_tag="$(basename "${meta_file}")"
  meta_tag="${meta_tag%.*}"
  infer_meta_root="${OUTPUT_BASE}/infer/${meta_tag}"

  if [[ ! -d "${infer_meta_root}" ]]; then
    echo "[WARN] infer output missing, skip CER for ${set_name}: ${infer_meta_root}" >&2
    return 0
  fi

  shopt -s nullglob
  for exp_dir in "${infer_meta_root}"/*; do
    [[ -d "${exp_dir}" ]] || continue
    local exp_name
    exp_name="$(basename "${exp_dir}")"
    for epoch_dir in "${exp_dir}"/epoch_*_whole; do
      [[ -d "${epoch_dir}" ]] || continue
      local epoch_name exp_tag result_dir wer_txt latency_tsv rc st meta_total paired cer_pct
      epoch_name="$(basename "${epoch_dir}")"
      exp_tag="${exp_name}/${epoch_name}"
      result_dir="${OUTPUT_BASE}/wer_scores/${meta_tag}/${exp_tag}/results"
      wer_txt="${result_dir}/combined_WER.txt"
      latency_tsv="${epoch_dir}/latency.tsv"

      echo "[opd-eval] CER ${set_name} ${exp_tag}"
      set +e
      META_FILE="${meta_file}" WAV_DIR="${epoch_dir}" EXP_TAG="${exp_tag}" \
        WER_OUT_ROOT="${OUTPUT_BASE}/wer_scores" \
        NUM_JOB="${WER_NUM_JOB}" GPU_OFFSET="${WER_GPU_OFFSET}" \
        bash "${MIDTERM_SCRIPT_DIR}/run_ceping_wer.sh"
      rc=$?
      set -e

      meta_total=""
      paired=""
      cer_pct=""
      [[ -f "${result_dir}/.meta_total" ]] && meta_total="$(cat "${result_dir}/.meta_total")"
      [[ -f "${result_dir}/.paired" ]] && paired="$(cat "${result_dir}/.paired")"
      [[ -f "${result_dir}/.cer_pct" ]] && cer_pct="$(cat "${result_dir}/.cer_pct")"
      if [[ "${rc}" -eq 0 ]]; then
        st="ok"
      else
        st="fail_${rc}"
      fi
      echo -e "${set_name}\t${meta_tag}\t${exp_name}\t${epoch_name}\t${meta_total}\t${paired}\t${cer_pct}\t${st}\t${wer_txt}\t${epoch_dir}\t${latency_tsv}" \
        >> "${summary_all}"
    done
  done
  shopt -u nullglob
}

if [[ "${RUN_INFER}" == "1" ]]; then
  run_infer_for_meta "${META_SET1}"
  run_infer_for_meta "${META_SET2}"
  run_infer_for_meta "${META_SET3}"
else
  echo "[opd-eval] Infer skipped (RUN_INFER=0)."
fi

if [[ "${SUMMARIZE_LATENCY}" == "1" ]]; then
  python3 "${REPO_ROOT}/examples/huawei_sft/summarize_opd_latency.py" \
    --infer-root "${OUTPUT_BASE}/infer" \
    --output "${LATENCY_SUMMARY_TSV}"
fi

if [[ "${RUN_CER}" == "1" ]]; then
  summary_all="${OUTPUT_BASE}/summary_all.tsv"
  echo -e "set\tmeta\texp\tepoch\tmeta_total\tpaired\tcer_pct\tstatus\twer_txt\twav_dir\tlatency_tsv" > "${summary_all}"
  run_cer_for_meta "set1" "${META_SET1}" "${summary_all}"
  run_cer_for_meta "set2" "${META_SET2}" "${summary_all}"
  run_cer_for_meta "set3" "${META_SET3}" "${summary_all}"
  cer_rows="$(awk 'NR > 1 {count++} END {print count + 0}' "${summary_all}")"
  cer_fail_rows="$(awk -F '\t' 'NR > 1 && $8 != "ok" {count++} END {print count + 0}' "${summary_all}")"
  if [[ "${REQUIRE_CER_ROWS}" == "1" && "${cer_rows}" -eq 0 ]]; then
    echo "[ERROR] no CER rows were produced; check infer outputs under ${OUTPUT_BASE}/infer" >&2
    exit 1
  fi
  if [[ "${REQUIRE_CER_OK}" == "1" && "${cer_fail_rows}" -gt 0 ]]; then
    echo "[ERROR] ${cer_fail_rows} CER rows failed; inspect ${summary_all}" >&2
    exit 1
  fi
  python3 "${MIDTERM_SCRIPT_DIR}/aggregate_cer_summary.py" "${summary_all}"
  if [[ "${COMPARE_EVAL}" == "1" ]]; then
    python3 "${REPO_ROOT}/examples/huawei_sft/compare_opd_eval.py" \
      --summary-all "${summary_all}" \
      --latency-summary "${LATENCY_SUMMARY_TSV}" \
      --output "${COMPARE_SUMMARY_TSV}"
    compare_rows="$(awk 'NR > 1 {count++} END {print count + 0}' "${COMPARE_SUMMARY_TSV}")"
    if [[ "${REQUIRE_COMPARE_ROWS}" == "1" && -n "${BASELINE_MODEL_DIRS}" && "${compare_rows}" -eq 0 ]]; then
      echo "[ERROR] no baseline-vs-OPD comparison rows were produced; inspect ${COMPARE_SUMMARY_TSV}" >&2
      exit 1
    fi
  fi
  echo "[opd-eval] CER summary -> ${summary_all}"
  echo "[opd-eval] CER matrix  -> ${OUTPUT_BASE}/summary_cer_matrix.tsv"
  if [[ "${COMPARE_EVAL}" == "1" ]]; then
    echo "[opd-eval] Compare TSV -> ${COMPARE_SUMMARY_TSV}"
  fi
else
  echo "[opd-eval] CER skipped (RUN_CER=0)."
fi
