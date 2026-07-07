#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs_8way_spk_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

SCRIPTS=(
  "run_sft_xiaoyuzhou_f03_5-5_1e-5_spk.sh"
  "run_sft_xiaoyuzhou_f03_5-5_1e-6_spk.sh"
  "run_sft_xiaoyuzhou_f03_10-5_1e-5_spk.sh"
  "run_sft_xiaoyuzhou_f03_10-5_1e-6_spk.sh"
  "run_sft_xiaoyuzhou_shenhu_5-5_1e-5_spk.sh"
  "run_sft_xiaoyuzhou_shenhu_5-5_1e-6_spk.sh"
  "run_sft_xiaoyuzhou_shenhu_10-5_1e-5_spk.sh"
  "run_sft_xiaoyuzhou_shenhu_10-5_1e-6_spk.sh"
)

echo "Log dir: ${LOG_DIR}"
echo "Launching ${#SCRIPTS[@]} SPK jobs in parallel..."

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

echo "All 8 SPK jobs completed successfully."
