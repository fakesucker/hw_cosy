#!/usr/bin/env bash
set -euo pipefail

# Parallel launcher for Shenhu filter-WER SPK SFT jobs (mirrors run_all_sft_8_parallel_top500k_spk.sh).
# GPU layout matches the underlying run_sft_xiaoyuzhou_shenhu_* bases (CUDA 0 + 4–7). Do not run
# together with run_all_sft_8_parallel_top500k_spk.sh on the same machine unless you change GPUs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs_5way_shenhu_filter_wer_spk_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

SCRIPTS=(
  "run_sft_shenhu_filter_wer_only_5-5_1e-5_spk.sh"
  "run_sft_top500k_shenhu_filter_wer_5-5_1e-5_spk.sh"
  "run_sft_top500k_shenhu_filter_wer_5-5_1e-6_spk.sh"
  "run_sft_top500k_shenhu_filter_wer_10-5_1e-5_spk.sh"
  "run_sft_top500k_shenhu_filter_wer_10-5_1e-6_spk.sh"
)

echo "Log dir: ${LOG_DIR}"
echo "Launching ${#SCRIPTS[@]} Shenhu filter-WER SPK jobs in parallel..."

pids=()
for script in "${SCRIPTS[@]}"; do
  if [[ ! -f "${SCRIPT_DIR}/${script}" ]]; then
    echo "[ERROR] Missing script: ${SCRIPT_DIR}/${script}"
    exit 1
  fi

  log_file="${LOG_DIR}/${script%.sh}.log"
  echo "[START] ${script} -> ${log_file}"
  (
    cd "${SCRIPT_DIR}"
    bash "${script}"
  ) >"${log_file}" 2>&1 &
  pids+=($!)
done

echo "All jobs submitted."
echo "Use: tail -f ${LOG_DIR}/*.log"

fail=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  script="${SCRIPTS[$i]}"
  if wait "${pid}"; then
    echo "[DONE] ${script}"
  else
    echo "[FAIL] ${script}"
    fail=1
  fi
done

if [[ "${fail}" -ne 0 ]]; then
  echo "Some jobs failed. Check logs in: ${LOG_DIR}"
  exit 1
fi

echo "All ${#SCRIPTS[@]} Shenhu filter-WER SPK jobs completed successfully."
