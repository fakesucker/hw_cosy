#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/run_sft_xiaoyuzhou_f03_10-5_1e-5.sh"
TMP_SCRIPT="$(mktemp /tmp/run_sft_xiaoyuzhou_f03_10-5_1e-5_spk.XXXXXX.sh)"
trap 'rm -f "${TMP_SCRIPT}"' EXIT

sed \
  -e 's#conf/cosyvoice2_sft_1e-5.yaml#conf/cosyvoice2_sft_1e-5_spk.yaml#g' \
  -e 's#sft_xiaoyuzhou_f03_10-5_1e-5#sft_xiaoyuzhou_f03_10-5_1e-5_spk#g' \
  "${BASE_SCRIPT}" > "${TMP_SCRIPT}"

bash "${TMP_SCRIPT}"
