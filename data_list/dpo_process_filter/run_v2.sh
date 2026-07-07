export GEMINI_API_KEY=sk-Eu7adosdYbKT9tgRjOyC4Ls5GHyWsTRF4DGeO5TnBpChkNN6
export GEMINI_URL=https://apim1tocn.cheapapi.ai

# python3 /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/gemini_understanding/analyse_for_kefu_v2.py \
#   --input_jsonl /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo/grouped_speech_tokens/dpo_rows_with_group_token_dedup.jsonl \
#   --output_jsonl /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/gemini_understanding/kefu_dpo_pairs_v2.jsonl \
#   --discard_log /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/gemini_understanding/kefu_dpo_pairs_v2_discard.log \
#   --summary_json /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/gemini_understanding/kefu_dpo_pairs_v2_summary.json \
#   --model gemini-3.1-pro-preview \
#   --concurrent_groups 1 \
#   --min_group_size 2 \
#   --max_retries 3 \
#   --sleep_between_calls 0.3


# python3 /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/gemini_understanding/analyse_for_kefu_v2.py \
#   --input_jsonl /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/testset_50.jsonl \
#   --out_dir /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/gemini_understanding/out_test50 \
#   --model gemini-3.1-pro-preview \
#   --concurrent_groups 1 \
#   --min_group_size 1 \
#   --max_retries 3 \
#   --sleep_between_calls 0.3 \
#   --save_win_lose_audio \
#   --audio_save_mode symlink


# python3 /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/gemini_understanding/analyse_for_kefu_v2.py \
#   --input_jsonl /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/testset_group_50.jsonl \
#   --input_format grouped \
#   --out_dir /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/gemini_understanding/out_grouped_test \
#   --model gemini-3.1-pro-preview \
#   --concurrent_groups 1 \
#   --min_group_size 2 \
#   --max_retries 3 \
#   --sleep_between_calls 0.3 \
#   --save_win_lose_audio \
#   --audio_save_mode symlink


python3 /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/gemini_understanding/analyse_for_kefu_v2.py \
  --input_jsonl /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo/v2_test/prepared_out/03_regrouped/dpo_grouped_by_prefix_token_dedup.jsonl \
  --input_format grouped \
  --out_dir /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/gemini_understanding/out_grouped_v2_test \
  --model gemini-3.1-pro-preview \
  --concurrent_groups 1 \
  --min_group_size 2 \
  --max_retries 3 \
  --sleep_between_calls 0.3 \
  --save_win_lose_audio \
  --audio_save_mode symlink

/home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo/v2_0428/prepared_out/gemini_sample_3k10/grouped_sampled_for_gemini.jsonl