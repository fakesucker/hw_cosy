#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
# SFT LLM: 4-GPU, bistream mode, text:speech mix_ratio 5:10.
# Override via env (used by run_all_sft_init_ckpt_seq_4gpu_bigbatch_stream_5_10.sh):
#   INIT_SFT_CKPT, SFT_OUTPUT_DIR, TRAIN_JOB_ID, TRAIN_BRANCH_MODE, CUDA_VISIBLE_DEVICES
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=/home/work_nfs23/hkxie/hw_proj/CosyVoice:/home/work_nfs23/hkxie/hw_proj/CosyVoice/third_party/Matcha-TTS:$PYTHONPATH

stage=5
stop_stage=5
pretrained_model_dir=/home/work_nfs23/hkxie/code/CosyVoice/pretrained_models/CosyVoice2-0.5B

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id="${TRAIN_JOB_ID:-2026}"
rdzv_port="${TRAIN_RDZV_PORT:-12459}"
dist_backend="nccl"
num_workers=1
prefetch=200
train_engine=torch_ddp
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Run train. We only support llm traning for now"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi

  train_branch_mode="${TRAIN_BRANCH_MODE:-bistream}"
  train_config="conf/cosyvoice2_sft_1e-6_spk_bistream_5_10.yaml"
  train_list="/home/work_nfs23/hkxie/hw_proj/CosyVoice/data_list/splits/top500k_shenhu_10-5_bistream.lst"
  cv_list="/home/work_nfs23/hkxie/data/kefu_zhongdu/shenhu/yuzhou_caption_spkemb400w.jsonl_test_splits.list"
  init_sft_ckpt="${INIT_SFT_CKPT:-/home/work_nfs23/hkxie/ckpt/huawei/cosyvoice2/dpo_xiaoyuzhou_shenhu_10-5_1e-6_bigbatch/epoch_5_whole.pt}"
  output_dir="${SFT_OUTPUT_DIR:-/home/node62_data/hkxie/ckpt/huawei/cosyvoice2/sft_xiaoyuzhou_shenhu_10-5_1e-6_bistream_4gpu_5_10_bigbatch}"

  echo "[NOTE] 4-GPU SFT: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, num_gpus=${num_gpus}"
  echo "[NOTE] SFT LLM train_branch_mode=${train_branch_mode}, mix_ratio=[5, 10] (text:speech)"
  echo "✅ Using init_sft_ckpt: ${init_sft_ckpt}"

  if [[ ! -f "${init_sft_ckpt}" ]]; then
    echo "❌ init_sft_ckpt not found: ${init_sft_ckpt}"
    exit 1
  fi

  for model in llm; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:${rdzv_port}" \
      ../../cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config ${train_config} \
      --train_data ${train_list} \
      --cv_data ${cv_list} \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --onnx_path $pretrained_model_dir \
      --model $model \
      --checkpoint ${init_sft_ckpt} \
      --model_dir ${output_dir} \
      --tensorboard_dir ${output_dir}/log \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer \
      --train_branch_mode "${train_branch_mode}"
  done
fi

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
