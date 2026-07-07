#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
# . ./path.sh || exit 1;
# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=/home/work_nfs23/hkxie/hw_proj/CosyVoice:/home/work_nfs23/hkxie/hw_proj/CosyVoice/third_party/Matcha-TTS:$PYTHONPATH

stage=5
stop_stage=5
# source /home/environment3/xmren/miniconda3/bin/activate /home/environment3/xmren/miniconda3/envs/cosyvoice/
pretrained_model_dir=/home/work_nfs23/hkxie/code/CosyVoice/pretrained_models/CosyVoice2-0.5B

export HF_ENDPOINT=https://hf-mirror.com
# train llm
# export CUDA_VISIBLE_DEVICES="5"
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES="5"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1988
dist_backend="nccl"
num_workers=1
prefetch=200
# prefetch=0
train_engine=torch_ddp
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Run train. We only support llm traning for now"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  # use yuzhou's shenhu checkpoint
  ckpt_dir="/home/work_nfs23/hkxie/code/CosyVoice/examples/huawei/cosyvoice2/exp/cosyvoice2/llm/stack_streaming_with_spk_norm_1201/torch_ddp"

  # === 1. 优先查找 epoch_*_step_*.pt ===
  latest_ckpt=$(find "$ckpt_dir" -type f -name "epoch_*_step_*.pt" 2>/dev/null \
    | sed -E 's/.*epoch_([0-9]+)_step_([0-9]+)\.pt/\1 \2 \0/' \
    | sort -k1,1nr -k2,2nr \
    | head -n1 \
    | awk '{print $3}')

  # # === 2. 如果找不到 step ckpt，则 fallback 到 epoch_*_whole.pt ===
  # if [[ -z "$latest_ckpt" ]]; then
  #   echo "⚠ No epoch_step checkpoint found, fallback to epoch_whole..."

  #   latest_ckpt=$(find "$ckpt_dir" -type f -name "epoch_*_whole.pt" 2>/dev/null \
  #     | sed -E 's/.*epoch_([0-9]+)_whole\.pt/\1 \0/' \
  #     | sort -k1,1nr \
  #     | head -n1 \
  #     | awk '{print $2}')
  # fi

  # # === 3. Final check ===
  # if [[ -z "$latest_ckpt" ]]; then
  #   echo "❌ No checkpoint found in: $ckpt_dir"
  #   exit 1
  # fi

  echo "✅ Using checkpoint: $latest_ckpt"

  # /home/work_nfs23/hkxie/ckpt/huawei/$model/stack_streaming_with_campplus/$train_engine
  # NOTE will update llm/hift training later
  for model in llm; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:12456" \
      ../../cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice2_sft_1e-6.yaml \
      --train_data /home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/splits/xiaoyuzhou_shenhu_5-5.lst \
      --cv_data /home/work_nfs23/hkxie/data/kefu_zhongdu/shenhu/yuzhou_caption_spkemb400w.jsonl_test_splits.list \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --onnx_path $pretrained_model_dir \
      --model $model \
      --checkpoint /home/work_nfs23/xmren/CosyVoice/examples/libritts/cosyvoice2/exp/shenhu_ckpt/epoch_0_step_40000.pt \
      --model_dir /home/work_nfs23/hkxie/ckpt/huawei/cosyvoice2/sft_xiaoyuzhou_shenhu_5-5_1e-6 \
      --tensorboard_dir /home/work_nfs23/hkxie/ckpt/huawei/cosyvoice2/sft_xiaoyuzhou_shenhu_5-5_1e-6/log \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
  done
fi

# average model
average_num=5
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  for model in llm flow hifigan; do
    decode_checkpoint=`pwd`/exp/cosyvoice/$model/$train_engine/${model}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path `pwd`/exp/cosyvoice/$model/$train_engine  \
      --num ${average_num} \
      --val_best
  done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
  python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
  python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
fi
