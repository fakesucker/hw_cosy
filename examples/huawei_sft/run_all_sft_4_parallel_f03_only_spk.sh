#!/usr/bin/env bash
set -euo pipefail

# Parallel launcher for F03-only SPK SFT jobs (mirrors run_all_sft_5_parallel_shenhu_filter_wer_spk.sh).
# Trains only on F03_10k.lst; four experiments in parallel:
#   - shenhu ckpt + lr 1e-5 / 1e-6
#   - CosyVoice2 base llm.pt + lr 1e-5 / 1e-6
# GPU layout (physical IDs): 0, 1, 2, 3. Do not run together with other parallel
# launchers on the same machine unless you change CUDA_VISIBLE_DEVICES in the child scripts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs_4way_f03_only_spk_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

SCRIPTS=(
  "run_sft_f03_5-5_1e-5_spk.sh"
  "run_sft_f03_5-5_1e-6_spk.sh"
  "run_sft_f03_base_1e-5_spk.sh"
  "run_sft_f03_base_1e-6_spk.sh"
)

echo "Log dir: ${LOG_DIR}"
echo "Launching ${#SCRIPTS[@]} F03-only SPK jobs in parallel..."
echo "  GPU 0: run_sft_f03_5-5_1e-5_spk.sh    (shenhu ckpt, lr=1e-5)"
echo "  GPU 1: run_sft_f03_5-5_1e-6_spk.sh    (shenhu ckpt, lr=1e-6)"
echo "  GPU 2: run_sft_f03_base_1e-5_spk.sh   (base llm.pt, lr=1e-5)"
echo "  GPU 3: run_sft_f03_base_1e-6_spk.sh   (base llm.pt, lr=1e-6)"

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

echo "All ${#SCRIPTS[@]} F03-only SPK jobs completed successfully."
