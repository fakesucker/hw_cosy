#!/bin/bash

########################################
# 1. 激活 conda 环境
########################################
# source /home/environment2/hkxie/anaconda3/bin/activate \
#        /home/environment2/hkxie/anaconda3/envs/s3token

source /home/environment2/hkxie/anaconda3/bin/activate /home/environment2/hkxie/anaconda3/envs/s3token

########################################
# 2. 设置 HuggingFace 国内镜像
########################################
# Use HF mirror (recommended for CN network)
export HF_ENDPOINT="https://hf-mirror.com"      # 主镜像
# export HUGGINGFACE_HUB_CACHE="/home/work_nfs19/hkxie/.cache/huggingface"
# export TRANSFORMERS_CACHE="/home/work_nfs19/hkxie/.cache/huggingface/transformers"
# export HF_HOME="/home/work_nfs19/hkxie/.cache/huggingface"

mkdir -p $HUGGINGFACE_HUB_CACHE
mkdir -p $TRANSFORMERS_CACHE

########################################
# 3. 限制当前任务使用 GPU
########################################
export CUDA_VISIBLE_DEVICES=4,5,6,7

########################################
# 4. 打印环境信息（可选，但建议保留）
########################################
echo "Using CUDA: $CUDA_VISIBLE_DEVICES"
echo "HF Mirror: $HF_ENDPOINT"
echo "Conda Env: $(which python)"
echo "Torch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "=========================================="

########################################
# 5. 启动分布式（单机） s3tokenizer
########################################
torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --rdzv_id=2024 \
    --rdzv_backend="c10d" \
    --rdzv_endpoint="localhost:6789" \
    $(which s3tokenizer) \
        --wav_scp /home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/prompt_wav/formatted_answer.scp \
        --device "cuda" \
        --output_dir "/home/work_nfs23/hkxie/huawei_streaming_cosyvoice/huawei_streaming_cosyvoice/kefu_test/s3tokens/" \
        --batch_size 10 \
        --model "speech_tokenizer_v2_25hz" \

########################################
echo "==== s3tokenizer job finished ===="