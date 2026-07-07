/home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo/v2_0428/prepared_out/03_regrouped/dpo_grouped_by_prefix_token_dedup.jsonl

python3 /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo/sample_grouped_for_gemini.py \
  --input_jsonl /path/to/your_grouped.jsonl \
  --out_dir /path/to/out_dir \
  --sample_groups 3000 \
  --sample_items_per_group 10 \
  --required_group_size 30 \
  --seed 42