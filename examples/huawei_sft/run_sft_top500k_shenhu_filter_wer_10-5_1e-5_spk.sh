#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/run_sft_xiaoyuzhou_shenhu_10-5_1e-5.sh"
TMP_SCRIPT="$(mktemp /tmp/run_sft_top500k_shenhu_filter_wer_10-5_1e-5_spk.XXXXXX.sh)"
trap 'rm -f "${TMP_SCRIPT}"' EXIT

sed \
  -e 's#conf/cosyvoice2_sft_1e-5.yaml#conf/cosyvoice2_sft_1e-5_spk.yaml#g' \
  -e 's#/home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/splits/[^ ]*#/home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/splits/top500k_shenhu_filter_wer_10-5.lst#g' \
  -e 's#sft_xiaoyuzhou_shenhu_10-5_1e-5#sft_top500k_shenhu_filter_wer_10-5_1e-5_spk#g' \
  "${BASE_SCRIPT}" > "${TMP_SCRIPT}"

bash "${TMP_SCRIPT}"
