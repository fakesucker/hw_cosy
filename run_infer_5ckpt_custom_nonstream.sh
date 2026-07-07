#!/usr/bin/env bash
#
# run_infer_5ckpt_custom_nonstream.sh
# 基于 run_all_train_infer_summary_4gpu_bistream_5_10.sh 改造：
#   - 跳过 Phase 1（训练）
#   - Phase 2 替换为 5 个自定义 checkpoint 的**非流式整句**推理
#   - 跳过 Phase 3/4（可选开启）
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
MIDTERM_SCRIPT_DIR="/home/work_nfs23/hkxie/hw_proj/testset_midterm/srcipt"

source "${MIDTERM_SCRIPT_DIR}/config.sh"

# --- phase switches (only Phase 2 by default) ---
RUN_TRAIN="${RUN_TRAIN:-0}"
RUN_INFER="${RUN_INFER:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-0}"
RUN_CONCAT="${RUN_CONCAT:-0}"

# --- shared paths ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,5}"
GPU_LIST="${GPU_LIST:-0,1,2,3,5}"
BASE_MODEL_DIR="/home/work_nfs24/xmren/Cosyvoice2-0.5B"
META_FILE="${META_FILE:-/home/work_nfs23/hkxie/hw_proj/testset_midterm/cmos/dialogue.lst}"
OUTPUT_BASE="${REPO_ROOT}/testout/ckpt_comparison_5_nonstream"
META_TAG="$(basename "${META_FILE}")"
META_TAG="${META_TAG%.*}"
CONCAT_OUT="${CONCAT_OUT:-${OUTPUT_BASE}/dialog_mixed_${META_TAG}_preset_user_8k}"
CONCAT_TOOL="${CONCAT_TOOL:-/home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/tools/concat_dialog_with_fixed_female.py}"
USER_AUDIO_MANIFEST="${USER_AUDIO_MANIFEST:-${FEMALE_MANIFEST:-/home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/simalution_DIA_female_manifest.tsv}}"
CONCAT_INPUT_DIR="${CONCAT_INPUT_DIR:-${OUTPUT_BASE}/concat_inputs_${META_TAG}}"
CONCAT_META_FILE="${CONCAT_META_FILE:-${CONCAT_INPUT_DIR}/${META_TAG}_with_preset_user.lst}"
CONCAT_USER_MANIFEST="${CONCAT_USER_MANIFEST:-${CONCAT_INPUT_DIR}/${META_TAG}_preset_user_manifest.tsv}"
CONCAT_OUTPUT_SAMPLE_RATE="${CONCAT_OUTPUT_SAMPLE_RATE:-8000}"

CONDA_SH="${CONDA_SH:-/home/environment2/hkxie/anaconda3/bin/activate}"
CONDA_ENV_INFER="${CONDA_ENV_INFER:-/home/environment2/hkxie/anaconda3/envs/cosyvoice2}"

timestamp() { date '+%F %T'; }

# --- 5 custom checkpoints (name ckpt_path gpu) ---
# 格式: "name|ckpt_path|gpu_id"
CUSTOM_CKPTS=(
  "sft_only_epoch3|/home/work_nfs23/hkxie/ckpt/huawei/cosyvoice2/sft_shenhu_filter_wer_only_1e-5_from_llm/epoch_3_whole.pt|0"
  "dpo_epoch5|/home/work_nfs23/hkxie/ckpt/huawei/cosyvoice2/dpo_xiaoyuzhou_shenhu_10-5_1e-6_bigbatch/epoch_5_whole.pt|1"
  "opd_turn1_epoch6_step78540|/home/work_nfs23/hkxie/hw_proj/CosyVoice/examples/huawei_sft/exp/cosyvoice2/opd_distill_opsd_topk16_dialogue_shenhu_fixed_turn_010_prompt_dpo5_nonstream_mf4000_cuda4_5_6_7_premium_male10_female10_20260630/torch_ddp/epoch_6_step_78540.pt|2"
  "opd_turn2_epoch6_step78540|/home/work_nfs23/hkxie/hw_proj/CosyVoice/examples/huawei_sft/exp/cosyvoice2/opd_distill_opsd_topk16_dialogue_shenhu_fixed_turn_010_prompt_dpo5_nonstream_mf4000_cuda4_5_6_7_full_20260629_cuda4_7/torch_ddp/epoch_6_step_78540.pt|3"
  "grpo_epoch0_step100|/home/node62_data/hkxie/ckpt/huawei/grpo/epoch_0_step_100/llm.pt|5"
)

echo "============================================================"
echo "[pipeline] $(timestamp) 5ckpt NON-STREAM inference"
echo "[pipeline] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[pipeline] META_FILE=${META_FILE}"
echo "[pipeline] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[pipeline] CONCAT_OUT=${CONCAT_OUT}"
echo "[pipeline] CONCAT_OUTPUT_SAMPLE_RATE=${CONCAT_OUTPUT_SAMPLE_RATE}"
echo "[pipeline] Mode: whole-text (NO --stream, NO --bistream_fixed_ratio)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Phase 1: Training (SKIPPED)
# ---------------------------------------------------------------------------
echo "[pipeline] Phase 1 skipped (RUN_TRAIN=0)."

# ---------------------------------------------------------------------------
# Phase 2: Inference (5 custom ckpts, non-stream, parallel on 5 GPUs)
# ---------------------------------------------------------------------------
if [[ "${RUN_INFER}" == "1" ]]; then
  echo ""
  echo "========== Phase 2: 5-Ckpt NON-STREAM Inference =========="
  source "${CONDA_SH}" "${CONDA_ENV_INFER}"
  cd "${REPO_ROOT}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export COSYVOICE_CUSTOMER_SERVICE_TN="${COSYVOICE_CUSTOMER_SERVICE_TN:-1}"

  mkdir -p "${OUTPUT_BASE}"

  declare -A infer_pids=()

  run_one_ckpt() {
    local name="$1"
    local ckpt="$2"
    local gpu_id="$3"
    local out_dir="${OUTPUT_BASE}/${name}"

    mkdir -p "${out_dir}"
    echo "[${name}] [$(timestamp)] START on GPU ${gpu_id} (NON-STREAM)"

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
      echo "[${name}] [$(timestamp)] DONE (ok)"
      echo "ok" > "${out_dir}/.status"
    else
      echo "[${name}] [$(timestamp)] DONE (FAIL, exit=${rc})"
      echo "fail_${rc}" > "${out_dir}/.status"
    fi
    return $rc
  }

  for entry in "${CUSTOM_CKPTS[@]}"; do
    IFS='|' read -r name ckpt gpu_id <<< "${entry}"
    run_one_ckpt "${name}" "${ckpt}" "${gpu_id}" &
    infer_pids["${name}"]=$!
    echo "[pipeline] Launched ${name} (PID=${infer_pids[${name}]}) on GPU ${gpu_id}"
  done

  echo ""
  echo "[pipeline] All 5 jobs launched. Waiting..."
  echo ""

  overall=0
  for name in "${!infer_pids[@]}"; do
    pid="${infer_pids[${name}]}"
    if ! wait "${pid}"; then
      echo "[pipeline] ${name} FAILED"
      overall=1
    else
      echo "[pipeline] ${name} OK"
    fi
  done

  if [[ $overall -ne 0 ]]; then
    echo "[pipeline] Phase 2 done with SOME FAILURES (check logs)."
  else
    echo "[pipeline] Phase 2 done (all ok)."
  fi
else
  echo "[pipeline] Phase 2 skipped (RUN_INFER=0)."
fi

# ---------------------------------------------------------------------------
# Phase 3: CER summary (SKIPPED by default, set RUN_SUMMARY=1 to enable)
# ---------------------------------------------------------------------------
if [[ "${RUN_SUMMARY}" == "1" ]]; then
  echo ""
  echo "========== Phase 3: CER Summary =========="
  # ... (same as original, omitted for now)
  echo "[pipeline] Phase 3 done."
else
  echo "[pipeline] Phase 3 skipped (RUN_SUMMARY=0)."
fi

# ---------------------------------------------------------------------------
# Phase 4: Concat dialogue (preset user audio + inferred kefu audio, 8k output)
# ---------------------------------------------------------------------------
if [[ "${RUN_CONCAT}" == "1" ]]; then
  echo ""
  echo "========== Phase 4: Concat Dialogue (output ${CONCAT_OUTPUT_SAMPLE_RATE} Hz) =========="

  if [[ ! -f "${CONCAT_TOOL}" ]]; then
    echo "[WARN] CONCAT_TOOL not found, skip: ${CONCAT_TOOL}"
  elif [[ ! -f "${USER_AUDIO_MANIFEST}" ]]; then
    echo "[WARN] USER_AUDIO_MANIFEST not found, skip concat: ${USER_AUDIO_MANIFEST}"
  else
    source "${CONDA_SH}" "${CONDA_ENV_INFER}"
    cd "${REPO_ROOT}"
    mkdir -p "${CONCAT_OUT}" "${CONCAT_INPUT_DIR}"

    META_FILE="${META_FILE}" \
    USER_AUDIO_MANIFEST="${USER_AUDIO_MANIFEST}" \
    CONCAT_META_FILE="${CONCAT_META_FILE}" \
    CONCAT_USER_MANIFEST="${CONCAT_USER_MANIFEST}" \
    python - <<'__CONCAT_INPUT_PY__'
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
__CONCAT_INPUT_PY__

    male_roots=()
    for entry in "${CUSTOM_CKPTS[@]}"; do
      IFS='|' read -r name ckpt gpu_id <<< "${entry}"
      out_dir="${OUTPUT_BASE}/${name}"
      if [[ -d "${out_dir}" ]]; then
        male_roots+=("${out_dir}")
      else
        echo "[WARN] missing infer output dir for concat: ${out_dir}"
      fi
    done

    if [[ "${#male_roots[@]}" -eq 0 ]]; then
      echo "[WARN] no infer output dirs found, skip concat."
    else
      python "${CONCAT_TOOL}" \
        --kefu_list "${CONCAT_META_FILE}" \
        --female_manifest "${CONCAT_USER_MANIFEST}" \
        --male_root "${male_roots[@]}" \
        --epoch-glob "" \
        --hop-glob "" \
        --workspace-root "${REPO_ROOT}" \
        --output_root "${CONCAT_OUT}" \
        --output_sample_rate "${CONCAT_OUTPUT_SAMPLE_RATE}"
      echo "[pipeline] Phase 4 done. Concat output -> ${CONCAT_OUT} (${CONCAT_OUTPUT_SAMPLE_RATE} Hz)"
    fi
  fi
else
  echo "[pipeline] Phase 4 skipped (RUN_CONCAT=0)."
fi

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "[pipeline] $(timestamp) ALL DONE"
echo "============================================================"
echo ""
echo "=== RESULTS ==="
for entry in "${CUSTOM_CKPTS[@]}"; do
  IFS='|' read -r name ckpt gpu_id <<< "${entry}"
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
echo "  ${CONCAT_OUT}/                              <- concat dialog mixed output (${CONCAT_OUTPUT_SAMPLE_RATE} Hz, when RUN_CONCAT=1)"
echo "============================================================"
