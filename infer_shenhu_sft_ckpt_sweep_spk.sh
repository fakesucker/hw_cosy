#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/infer_shenhu_sft_ckpt_sweep.sh"

# SPK training dedicated inference defaults.
export CKPT_ROOT="${CKPT_ROOT:-/home/work_nfs23/hkxie/ckpt/huawei/cosyvoice2}"
export MODEL_DIRS_OVERRIDE="${MODEL_DIRS_OVERRIDE:-}"
export IS_SFT="${IS_SFT:-1}"
export SFT_SPK_ID="${SFT_SPK_ID:-中文女}"
export IS_USE_SPK_TAG="${IS_USE_SPK_TAG:-1}"
export SPK_TAG="${SPK_TAG:-<|spk_1|>}"
export TOP_N="${TOP_N:-5}"
export META_FILE="${META_FILE:-/home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/kefu_0423_onlymale.lst}"

if [[ -z "${MODEL_DIRS_OVERRIDE}" ]]; then
  # shellcheck disable=SC2016
  TMP_SCRIPT="$(mktemp /tmp/infer_shenhu_sft_ckpt_sweep_spk.XXXXXX.sh)"
  trap 'rm -f "${TMP_SCRIPT}"' EXIT
  sed \
    -e 's#sft_xiaoyuzhou_f03_5-5_1e-5#sft_xiaoyuzhou_f03_5-5_1e-5_spk#g' \
    -e 's#sft_xiaoyuzhou_f03_5-5_1e-6#sft_xiaoyuzhou_f03_5-5_1e-6_spk#g' \
    -e 's#sft_xiaoyuzhou_f03_10-5_1e-5#sft_xiaoyuzhou_f03_10-5_1e-5_spk#g' \
    -e 's#sft_xiaoyuzhou_f03_10-5_1e-6#sft_xiaoyuzhou_f03_10-5_1e-6_spk#g' \
    -e 's#sft_xiaoyuzhou_shenhu_5-5_1e-5#sft_xiaoyuzhou_shenhu_5-5_1e-5_spk#g' \
    -e 's#sft_xiaoyuzhou_shenhu_5-5_1e-6#sft_xiaoyuzhou_shenhu_5-5_1e-6_spk#g' \
    -e 's#sft_xiaoyuzhou_shenhu_10-5_1e-5#sft_xiaoyuzhou_shenhu_10-5_1e-5_spk#g' \
    -e 's#sft_xiaoyuzhou_shenhu_10-5_1e-6#sft_xiaoyuzhou_shenhu_10-5_1e-6_spk#g' \
    "${BASE_SCRIPT}" > "${TMP_SCRIPT}"
  bash "${TMP_SCRIPT}"
else
  bash "${BASE_SCRIPT}"
fi
