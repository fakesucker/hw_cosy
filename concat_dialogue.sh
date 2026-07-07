source /home/environment2/hkxie/anaconda3/bin/activate /home/environment2/hkxie/anaconda3/envs/cosyvoice2


# /home/environment2/hkxie/anaconda3/envs/cosyvoice2/bin/python \
#     /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/tools/concat_dialog_with_fixed_female.py \
#     --kefu_list /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/kefu.lst \
#     --female_manifest /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/simalution_DIA_female_manifest.tsv \
#     --male_root /home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/kefu_0421_onlyhw_niren \
#     --output_root /home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/dialog_mixed_with_hw_niren

# /home/environment2/hkxie/anaconda3/envs/cosyvoice2/bin/python \
#     /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/tools/concat_dialog_with_fixed_female.py \
#     --kefu_list /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/kefu.lst \
#     --female_manifest /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/simalution_DIA_female_manifest.tsv \
#     --male_root /home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/kefu_0421_onlymale \
#     --output_root /home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/dialog_mixed_with_hw_niren

# /home/environment2/hkxie/anaconda3/envs/cosyvoice2/bin/python \
#     /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/tools/concat_dialog_with_fixed_female.py \
#     --kefu_list /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/kefu.lst \
#     --female_manifest /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/simalution_DIA_female_manifest.tsv \
#     --male_root /home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/kefu_0421_onlymale \
#     --output_root /home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/dialog_mixed_with_hw_niren


# /home/environment2/hkxie/anaconda3/envs/cosyvoice2/bin/python \
#     /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/tools/concat_dialog_with_fixed_female.py \
#     --kefu_list /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/kefu.lst \
#     --female_manifest /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/simalution_DIA_female_manifest.tsv \
#     --male_root /home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/kefu_0423_onlymale \
#     --output_root /home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/dialog_mixed_with_hw_niren



# echo "0430 concat flow sft + dpo"
# /home/environment2/hkxie/anaconda3/envs/cosyvoice2/bin/python \
#     /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/tools/concat_dialog_with_fixed_female.py \
#     --kefu_list /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/kefu.lst \
#     --female_manifest /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/simalution_DIA_female_manifest.tsv \
#     --male_root /home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/test_flowsft_0428/kefu_0421_onlymale \
#     --output_root /home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/dialog_mixed_with_hw_niren_0430


# echo "0506 concat flow sft + dpo only male"
# /home/environment2/hkxie/anaconda3/envs/cosyvoice2/bin/python \
#     /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/tools/concat_dialog_with_fixed_female.py \
#     --kefu_list /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/kefu.lst \
#     --female_manifest /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/simalution_DIA_female_manifest.tsv \
#     --male_root /home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/test_flowsft_0428/kefu_0506_onlymale \
#     --output_root /home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/dialog_mixed_with_hw_niren_0506_onlymale


# echo "0510 concat flow sft only male"
# /home/environment2/hkxie/anaconda3/envs/cosyvoice2/bin/python \
#     /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/tools/concat_dialog_with_fixed_female.py \
#     --kefu_list /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/kefu.lst \
#     --female_manifest /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/simalution_DIA_female_manifest.tsv \
#     --male_root /home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/test_flowsft_0428/kefu_0506_onlymale \
#     --output_root /home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/dialog_mixed_with_hw_niren_0510_onlymale

# echo "0512 concat flow sft only male streaming sentence mode"
# /home/environment2/hkxie/anaconda3/envs/cosyvoice2/bin/python \
#     /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/tools/concat_dialog_with_fixed_female.py \
#     --kefu_list /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/kefu.lst \
#     --female_manifest /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/simalution_DIA_female_manifest.tsv \
#     --male_root /home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/test_flowsft_0512_stream_sentence/kefu_0506_onlymale \
#     --output_root /home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/dialog_mixed_with_hw_niren_0512_stream_sentence


# HUAWEI_ROOT=/home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice
# MALE_ROOT="${HUAWEI_ROOT}/testout/cpt_nonmtp_tokenlevel_stream"
# OUT_ROOT=/home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/dialog_mixed_with_flashtts_streaming
# PYTHON=/home/environment2/hkxie/anaconda3/envs/cosyvoice2/bin/python
# TOOL="${HUAWEI_ROOT}/tools/concat_dialog_with_fixed_female.py"

# # Streaming 推理目录: <exp>/first16_hop25/*.wav（不是旧的 epoch_*_whole）
# MALE_SUBDIR=first16_hop25

# echo "0518 concat flow sft only male streaming sentence mode"
# echo "  male_root=${MALE_ROOT}"
# echo "  male_subdir=${MALE_SUBDIR}"

# # 先预览能发现哪些目录（可选，去掉 --list-only 即正式拼接）
# # ${PYTHON} "${TOOL}" --list-only \
# #     --kefu_list "${HUAWEI_ROOT}/kefu_test/kefu.lst" \
# #     --female_manifest "${HUAWEI_ROOT}/kefu_test/simalution_DIA_female_manifest.tsv" \
# #     --male_root "${MALE_ROOT}" \
# #     --male-subdir "${MALE_SUBDIR}"

# cd "${HUAWEI_ROOT}" || exit 1
# ${PYTHON} "${TOOL}" \
#     --kefu_list "${HUAWEI_ROOT}/kefu_test/kefu.lst" \
#     --female_manifest "${HUAWEI_ROOT}/kefu_test/simalution_DIA_female_manifest.tsv" \
#     --male_root "${MALE_ROOT}" \
#     --male-subdir "${MALE_SUBDIR}" \
#     --workspace-root "${HUAWEI_ROOT}" \
#     --output_root "${OUT_ROOT}"


echo "0523 concat flow f03 spk streaming sentence mode"
/home/environment2/hkxie/anaconda3/envs/cosyvoice2/bin/python \
    /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/tools/concat_dialog_with_fixed_female.py \
    --kefu_list /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/kefu.lst \
    --female_manifest /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/simalution_DIA_female_manifest.tsv \
    --male_root /home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/test_f03_0521/kefu_0521_onlymale_f03 \
    --output_root /home/work_nfs23/hkxie/hw_proj/CosyVoice/testout/dialog_mixed_with_f03_spk
