export GEMINI_API_KEY=sk-Eu7adosdYbKT9tgRjOyC4Ls5GHyWsTRF4DGeO5TnBpChkNN6
export GEMINI_URL=https://apim1tocn.cheapapi.ai

# 默认：每个 group 完成后立即 append 写入 jsonl（中断不丢已完成结果）
# 默认断点续跑：跳过 kefu_dpo_pairs.jsonl 里已有的 group_id
# 强制从头：加 --truncate_output
# 不续跑全量重跑：加 --no_resume

# python3 /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/gemini_understanding/analyse_for_kefu_v2_mp.py \
#   --input_jsonl /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo/v2_test/prepared_out/03_regrouped/dpo_grouped_by_prefix_token_dedup.jsonl \
#   --input_format grouped \
#   --out_dir /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/gemini_understanding/out_grouped_v2_0428_mp_debug \
#   --model gemini-3.1-pro-preview \
#   --num_processes 8 \
#   --worker_concurrent_groups 2 \
#   --max_retries 3 \
#   --sleep_between_calls 0.3 \
#   --save_win_lose_audio \
#   --audio_save_mode symlink

# python3 /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/gemini_understanding/analyse_for_kefu_v2_mp.py \
#   --input_jsonl /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo/v2_0428/prepared_out/gemini_sample_3k10/grouped_sampled_for_gemini.jsonl \
#   --input_format grouped \
#   --out_dir /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/gemini_understanding/out_grouped_v2_0428_3k10_mp \
#   --model gemini-3.1-pro-preview \
#   --num_processes 8 \
#   --worker_concurrent_groups 2 \
#   --max_retries 3 \
#   --sleep_between_calls 0.3 \
#   --save_win_lose_audio \
#   --audio_save_mode symlink



python3 /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/gemini_understanding/analyse_for_kefu_v2_mp.py \
  --input_jsonl /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo/v3_midterm/prepared_out/03_regrouped/dpo_grouped_by_prefix_token_dedup.jsonl \
  --input_format grouped \
  --out_dir /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/gemini_understanding/out_grouped_v2_midterm_mp \
  --model gemini-3.1-pro-preview \
  --num_processes 8 \
  --worker_concurrent_groups 2 \
  --max_retries 5 \
  --sleep_between_calls 0.3 \
  --save_win_lose_audio \
  --audio_save_mode symlink