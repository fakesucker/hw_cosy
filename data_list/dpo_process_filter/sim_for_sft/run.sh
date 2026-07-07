SV_DIR=/home/work_nfs23/xmren/data/shenhu/seed-tts-eval/thirdparty/UniSpeech/downstreams/speaker_verification \
# EXTRA_ARGS="--limit 1000 --outlier_top_frac 0.02 --do_tsne" \
# bash /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/sim_for_sft/cal_shenhu_outliers.sh \
#   /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo/grouped_speech_tokens/dpo_rows_with_group_token_dedup.jsonl \
#   /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/sim_for_sft/out_dpo_rows \
#   /home/work_nfs14/code/hkxie/TTS/seed-tts-eval-main/smos/ckpt/wavlm_large_finetune.pth

SV_DIR=/home/work_nfs23/xmren/data/shenhu/seed-tts-eval/thirdparty/UniSpeech/downstreams/speaker_verification \
EXTRA_ARGS="--reuse_cache --single_cluster_mode --outlier_top_frac 0.02 --do_tsne --tsne_annotate_topn 15" \
bash /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/sim_for_sft/cal_shenhu_outliers.sh \
  /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo/grouped_speech_tokens/dpo_rows_with_group_token_dedup.jsonl \
  /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/dpo_process_filter/sim_for_sft/out_dpo_rows_single_cluster \
  /home/work_nfs14/code/hkxie/TTS/seed-tts-eval-main/smos/ckpt/wavlm_large_finetune.pth