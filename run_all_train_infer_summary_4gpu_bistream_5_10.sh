#!/usr/bin/env bash
#
# run_all_train_infer_summary_4gpu_bistream_5_10.sh
# -------------------------------------------------
# 一键流水线：4 卡 bistream 5:10 训练 → 4 卡推理（全部 epoch ckpt）→ CER 汇总 + 对话拼接
#
# 默认 GPU 0-3；训练/推理/ASR 共用同一组卡（分阶段串行，不会同时占满）。
#
# 全流程（训练 + 推理 + 汇总）：
#   cd /home/work_nfs23/hkxie/hw_proj/CosyVoice && \
#   bash run_all_train_infer_summary_4gpu_bistream_5_10.sh
#
# 跳过训练（已有 ckpt，直接推理+汇总）：
#   RUN_TRAIN=0 bash run_all_train_infer_summary_4gpu_bistream_5_10.sh
#
# 仅训练：
#   RUN_INFER=0 RUN_SUMMARY=0 bash run_all_train_infer_summary_4gpu_bistream_5_10.sh
#
# 仅汇总（推理已完成）：
#   RUN_TRAIN=0 RUN_INFER=0 bash run_all_train_infer_summary_4gpu_bistream_5_10.sh
#
set -euo pipefail

REPO_ROOT="/home/work_nfs23/hkxie/hw_proj/CosyVoice"
SFT_DIR="${REPO_ROOT}/examples/huawei_sft"
MIDTERM_SCRIPT_DIR="/home/work_nfs23/hkxie/hw_proj/testset_midterm/srcipt"

# shellcheck source=/dev/null
source "${MIDTERM_SCRIPT_DIR}/config.sh"

# --- phase switches ---
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_INFER="${RUN_INFER:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
RUN_CONCAT="${RUN_CONCAT:-1}"

# --- shared paths ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
export CKPT_ROOT="${CKPT_ROOT:-/home/node62_data/hkxie/ckpt/huawei/cosyvoice2}"
META_FILE="${META_FILE:-/home/work_nfs23/hkxie/hw_proj/testset_midterm/cmos/dialogue.lst}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/testout/test_init_bistream_4gpu_5_10_stream}"
META_TAG="$(basename "${META_FILE}")"
META_TAG="${META_TAG%.*}"
RUN_OUTPUT_BASE="${OUTPUT_BASE}/${META_TAG}"
SUMMARY_ALL="${OUTPUT_BASE}/summary_all_${META_TAG}.tsv"
CONCAT_OUT="${CONCAT_OUT:-${OUTPUT_BASE}/dialog_mixed_${META_TAG}_preset_user}"

# 训练 max_epoch=10；TOP_N=999 表示推理全部 epoch_*_whole.pt
TOP_N="${TOP_N:-999}"
ONLY_EXP="${ONLY_EXP:-}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-0}"
SKIP_IF_OUTPUT_EXISTS="${SKIP_IF_OUTPUT_EXISTS:-0}"

CONCAT_TOOL="${CONCAT_TOOL:-/home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/tools/concat_dialog_with_fixed_female.py}"
# Pre-recorded user-side audio. Phase 4 remaps this manifest to the CMOS utt_ids
# in META_FILE, so concat uses preset user audio + current inferred kefu audio.
USER_AUDIO_MANIFEST="${USER_AUDIO_MANIFEST:-${FEMALE_MANIFEST:-/home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/simalution_DIA_female_manifest.tsv}}"
CONCAT_INPUT_DIR="${CONCAT_INPUT_DIR:-${OUTPUT_BASE}/concat_inputs_${META_TAG}}"
CONCAT_META_FILE="${CONCAT_META_FILE:-${CONCAT_INPUT_DIR}/${META_TAG}_with_preset_user.lst}"
CONCAT_USER_MANIFEST="${CONCAT_USER_MANIFEST:-${CONCAT_INPUT_DIR}/${META_TAG}_preset_user_manifest.tsv}"
WER_NUM_JOB="${WER_NUM_JOB:-4}"
WER_GPU_OFFSET="${WER_GPU_OFFSET:-0}"

CONDA_SH="${CONDA_SH:-/home/environment2/hkxie/anaconda3/bin/activate}"
CONDA_ENV_INFER="${CONDA_ENV_INFER:-/home/environment2/hkxie/anaconda3/envs/cosyvoice2}"

timestamp() { date '+%F %T'; }

echo "============================================================"
echo "[pipeline] $(timestamp) 4gpu bistream 5:10 train -> infer -> summary"
echo "[pipeline] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} GPU_LIST=${GPU_LIST}"
echo "[pipeline] CKPT_ROOT=${CKPT_ROOT}"
echo "[pipeline] META_FILE=${META_FILE}"
echo "[pipeline] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[pipeline] RUN_TRAIN=${RUN_TRAIN} RUN_INFER=${RUN_INFER} RUN_SUMMARY=${RUN_SUMMARY} RUN_CONCAT=${RUN_CONCAT}"
echo "[pipeline] TOP_N=${TOP_N} (all epoch_*_whole.pt)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Phase 1: Training (7 init ckpts, serial, 4 GPUs each)
# ---------------------------------------------------------------------------
if [[ "${RUN_TRAIN}" == "1" ]]; then
  echo ""
  echo "========== Phase 1: 4-GPU Training (bistream 5:10) =========="
  (
    cd "${SFT_DIR}"
    export CUDA_VISIBLE_DEVICES
    export CKPT_ROOT
    export TRAIN_BRANCH_MODE=bistream
    export CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL}"
    export SKIP_IF_OUTPUT_EXISTS="${SKIP_IF_OUTPUT_EXISTS}"
    bash run_all_sft_init_ckpt_seq_4gpu_bigbatch_stream_5_10.sh
  )
  echo "[pipeline] Phase 1 done."
else
  echo "[pipeline] Phase 1 skipped (RUN_TRAIN=0)."
fi

# ---------------------------------------------------------------------------
# Phase 2: Inference (7 exps on 4 GPUs, all epoch ckpts)
# ---------------------------------------------------------------------------
if [[ "${RUN_INFER}" == "1" ]]; then
  echo ""
  echo "========== Phase 2: 4-GPU Inference (all ckpts) =========="
  source "${CONDA_SH}" "${CONDA_ENV_INFER}"
  cd "${REPO_ROOT}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export COSYVOICE_CUSTOMER_SERVICE_TN="${COSYVOICE_CUSTOMER_SERVICE_TN:-1}"

  RUN_MODE=parallel \
  GPU_LIST="${GPU_LIST}" \
  PROCS_PER_GPU=1 \
  TOP_N="${TOP_N}" \
  EPOCH_RUN_MODE=serial \
  ONLY_EXP="${ONLY_EXP}" \
  META_FILE="${META_FILE}" \
  OUTPUT_BASE="${OUTPUT_BASE}" \
  CKPT_ROOT="${CKPT_ROOT}" \
  BISTREAM_FIXED_RATIO=1 \
  BISTREAM_TEXT_CHUNK_TOKENS=5 \
  BISTREAM_SPEECH_CHUNK_TOKENS=10 \
  bash "${REPO_ROOT}/infer_shenhu_sft_ckpt_sweep_init_bistream_4gpu_5_10_stream_up.sh"

  echo "[pipeline] Phase 2 done. Infer summary: ${RUN_OUTPUT_BASE}/summary_runs.tsv"
else
  echo "[pipeline] Phase 2 skipped (RUN_INFER=0)."
fi

# ---------------------------------------------------------------------------
# Phase 3: CER summary (all exp × epoch)
# ---------------------------------------------------------------------------
if [[ "${RUN_SUMMARY}" == "1" ]]; then
  echo ""
  echo "========== Phase 3: CER Summary (all exp × epoch) =========="

  if [[ ! -d "${RUN_OUTPUT_BASE}" ]]; then
    echo "[ERROR] Infer output not found: ${RUN_OUTPUT_BASE}"
    exit 1
  fi
  if [[ ! -f "${META_FILE}" ]]; then
    echo "[ERROR] META_FILE not found: ${META_FILE}"
    exit 1
  fi

  mkdir -p "$(dirname "${SUMMARY_ALL}")"
  echo -e "set\tmeta\texp\tepoch\tmeta_total\tpaired\tcer_pct\tstatus\twer_txt\twav_dir" > "${SUMMARY_ALL}"

  shopt -s nullglob
  for exp_dir in "${RUN_OUTPUT_BASE}"/*; do
    [[ -d "${exp_dir}" ]] || continue
    exp_name="$(basename "${exp_dir}")"
    if [[ -n "${ONLY_EXP}" ]] && [[ "${exp_name}" != *"${ONLY_EXP}"* ]]; then
      continue
    fi

    for epoch_dir in "${exp_dir}"/epoch_*_whole; do
      [[ -d "${epoch_dir}" ]] || continue
      epoch_name="$(basename "${epoch_dir}")"
      exp_tag="${exp_name}/${epoch_name}"
      result_dir="${OUTPUT_BASE}/wer_scores/${META_TAG}/${exp_tag}/results"
      wer_txt="${result_dir}/combined_WER.txt"

      echo "[$(timestamp)] CER ${exp_tag}"
      set +e
      META_FILE="${META_FILE}" WAV_DIR="${epoch_dir}" EXP_TAG="${exp_tag}" \
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

      if [[ $rc -eq 0 ]]; then
        st="ok"
      else
        st="fail_${rc}"
      fi
      echo -e "set1\t${META_TAG}\t${exp_name}\t${epoch_name}\t${meta_total}\t${paired}\t${cer_pct}\t${st}\t${wer_txt}\t${epoch_dir}" \
        >> "${SUMMARY_ALL}"
    done
  done
  shopt -u nullglob

  python3 "${MIDTERM_SCRIPT_DIR}/aggregate_cer_summary.py" "${SUMMARY_ALL}"

  echo "[pipeline] Phase 3 done."
  echo "[pipeline] CER summary_all -> ${SUMMARY_ALL}"
  echo "[pipeline] CER matrix      -> $(dirname "${SUMMARY_ALL}")/summary_cer_matrix.tsv"
else
  echo "[pipeline] Phase 3 skipped (RUN_SUMMARY=0)."
fi

# ---------------------------------------------------------------------------
# Phase 4: Concat dialogue (preset user audio + inferred kefu audio)
# ---------------------------------------------------------------------------
if [[ "${RUN_CONCAT}" == "1" ]]; then
  echo ""
  echo "========== Phase 4: Concat Dialogue =========="

  if [[ ! -f "${CONCAT_TOOL}" ]]; then
    echo "[WARN] CONCAT_TOOL not found, skip: ${CONCAT_TOOL}"
  elif [[ ! -d "${RUN_OUTPUT_BASE}" ]]; then
    echo "[WARN] RUN_OUTPUT_BASE missing, skip concat."
  elif [[ ! -f "${USER_AUDIO_MANIFEST}" ]]; then
    echo "[WARN] USER_AUDIO_MANIFEST not found, skip concat: ${USER_AUDIO_MANIFEST}"
  else
    source "${CONDA_SH}" "${CONDA_ENV_INFER}"
    mkdir -p "${CONCAT_OUT}" "${CONCAT_INPUT_DIR}"

    META_FILE="${META_FILE}" \
    USER_AUDIO_MANIFEST="${USER_AUDIO_MANIFEST}" \
    CONCAT_META_FILE="${CONCAT_META_FILE}" \
    CONCAT_USER_MANIFEST="${CONCAT_USER_MANIFEST}" \
    python - <<'PY'
from __future__ import annotations

import os
import re
from pathlib import Path

meta_file = Path(os.environ["META_FILE"])
user_audio_manifest = Path(os.environ["USER_AUDIO_MANIFEST"])
concat_meta_file = Path(os.environ["CONCAT_META_FILE"])
concat_user_manifest = Path(os.environ["CONCAT_USER_MANIFEST"])


def norm_text(text: str) -> str:
    text = re.sub(r"#\d+", "", text)
    text = text.replace("@", "")
    return re.sub(r"[，。？！、,.?！“”\"\s…]+", "", text)


meta_rows = [line for line in meta_file.read_text(encoding="utf-8").splitlines() if line.strip()]
user_rows = []
for line in meta_rows:
    cols = line.split("|")
    if len(cols) >= 5 and "用户话术" in cols[-1]:
        user_rows.append((line, cols[0].strip(), norm_text(cols[3])))

manifest_lines = [
    line
    for line in user_audio_manifest.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not manifest_lines:
    raise SystemExit(f"[ERROR] empty USER_AUDIO_MANIFEST: {user_audio_manifest}")

header = manifest_lines[0].split("\t")
try:
    wav_idx = header.index("wav_path")
except ValueError as exc:
    raise SystemExit(f"[ERROR] USER_AUDIO_MANIFEST missing wav_path column: {user_audio_manifest}") from exc
text_idx = header.index("text") if "text" in header else None

preset_rows = []
for line in manifest_lines[1:]:
    cols = line.split("\t")
    if len(cols) <= wav_idx:
        continue
    preset_rows.append((cols[wav_idx], norm_text(cols[text_idx]) if text_idx is not None and len(cols) > text_idx else ""))

paired = min(len(user_rows), len(preset_rows))
if paired == 0:
    raise SystemExit("[ERROR] no preset user audio rows can be paired with META_FILE")

for idx in range(paired):
    expected = preset_rows[idx][1]
    if expected and user_rows[idx][2] != expected:
        raise SystemExit(
            "[ERROR] preset user manifest order does not match META_FILE at "
            f"row {idx}: meta_utt={user_rows[idx][1]}"
        )

mapped_user_ids = {user_rows[idx][1] for idx in range(paired)}
complete_sessions = set()
all_user_ids_by_session: dict[str, set[str]] = {}
for _, utt_id, _ in user_rows:
    parts = utt_id.split("_")
    session = "_".join(parts[:2])
    all_user_ids_by_session.setdefault(session, set()).add(utt_id)
for session, ids in all_user_ids_by_session.items():
    if ids.issubset(mapped_user_ids):
        complete_sessions.add(session)

concat_user_manifest.parent.mkdir(parents=True, exist_ok=True)
with concat_user_manifest.open("w", encoding="utf-8") as f:
    f.write("utt_id\tsession_key\tturn\twav_path\ttext\n")
    for idx in range(paired):
        _, utt_id, _ = user_rows[idx]
        parts = utt_id.split("_")
        session = "_".join(parts[:2])
        if session not in complete_sessions:
            continue
        f.write(f"{utt_id}\t{session}\t{parts[2]}\t{preset_rows[idx][0]}\t\n")

concat_meta_file.parent.mkdir(parents=True, exist_ok=True)
with concat_meta_file.open("w", encoding="utf-8") as f:
    for line in meta_rows:
        utt_id = line.split("|", 1)[0].strip()
        session = "_".join(utt_id.split("_")[:2])
        if session in complete_sessions:
            f.write(line + "\n")

print(
    "[concat-input] preset_user_rows="
    f"{paired}/{len(user_rows)} complete_sessions={len(complete_sessions)} "
    f"meta={concat_meta_file} user_manifest={concat_user_manifest}"
)
PY

    python "${CONCAT_TOOL}" \
      --kefu_list "${CONCAT_META_FILE}" \
      --female_manifest "${CONCAT_USER_MANIFEST}" \
      --male_root "${RUN_OUTPUT_BASE}" \
      --output_root "${CONCAT_OUT}"
    echo "[pipeline] Phase 4 done. Concat output -> ${CONCAT_OUT}"
  fi
else
  echo "[pipeline] Phase 4 skipped (RUN_CONCAT=0)."
fi

echo ""
echo "============================================================"
echo "[pipeline] $(timestamp) ALL DONE"
echo "[pipeline] Train ckpts : ${CKPT_ROOT}/sft_xiaoyuzhou_init_*_bistream_4gpu_5_10_bigbatch"
echo "[pipeline] Infer wavs  : ${RUN_OUTPUT_BASE}/<exp>/epoch_*_whole/"
echo "[pipeline] Infer log   : ${RUN_OUTPUT_BASE}/summary_runs.tsv"
echo "[pipeline] CER table   : ${SUMMARY_ALL}"
echo "[pipeline] Dialog mix  : ${CONCAT_OUT}"
echo "============================================================"
