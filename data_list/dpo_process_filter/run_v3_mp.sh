#!/usr/bin/env bash
set -euo pipefail

export GEMINI_API_KEY=sk-Eu7adosdYbKT9tgRjOyC4Ls5GHyWsTRF4DGeO5TnBpChkNN6
: "${GEMINI_API_KEY:?Please export GEMINI_API_KEY before running run_v3_mp.sh}"
export GEMINI_URL=https://apim1tocn.cheapapi.ai

ROOT_DIR="/home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter"
INPUT_JSONL="/home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo/v3_midterm/prepared_out/03_regrouped/dpo_grouped_by_prefix_token_dedup.jsonl"
OUT_DIR="${ROOT_DIR}/gemini_understanding/out_grouped_v3_midterm_mp"

# 默认：每个 group 完成后立即 append 写入 jsonl（中断不丢已完成结果）
# 默认断点续跑：跳过 kefu_dpo_pairs.jsonl 里已有的 group_id
# 强制从头：加 --truncate_output
# 不续跑全量重跑：加 --no_resume

python3 "${ROOT_DIR}/gemini_understanding/analyse_for_kefu_v3_mp.py" \
  --input_jsonl "${INPUT_JSONL}" \
  --input_format grouped \
  --out_dir "${OUT_DIR}" \
  --model gemini-3.1-pro-preview \
  --num_processes 8 \
  --worker_concurrent_groups 2 \
  --max_batch_size 6 \
  --max_retries 5 \
  --sleep_between_calls 0.5 \
  --final_pair_judge_rounds 3 \
  --min_pair_judge_votes 2 \
  --min_pair_margin 1.0 \
  --min_pair_confidence 3.0 \
  --save_win_lose_audio \
  --audio_save_mode symlink



# python3 "${ROOT_DIR}/gemini_understanding/analyse_for_kefu_v3_mp.py" \
#   --input_jsonl "${INPUT_JSONL}" \
#   --input_format grouped \
#   --out_dir "${OUT_DIR}" \
#   --model gemini-2.5-pro \
#   --num_processes 2 \
#   --worker_concurrent_groups 2 \
#   --max_batch_size 6 \
#   --max_retries 5 \
#   --sleep_between_calls 0.3 \
#   --final_pair_judge_rounds 3 \
#   --min_pair_judge_votes 2 \
#   --min_pair_margin 1.0 \
#   --min_pair_confidence 3.0 \
#   --save_win_lose_audio \
#   --audio_save_mode symlink
