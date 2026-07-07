#!/bin/bash
set -euo pipefail

export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../:../../third_party/Matcha-TTS:${PYTHONPATH:-}

# Usage:
#   TEACHER_CHECKPOINT=/path/to/teacher.pt bash run_opd_distill_llm.sh
# Optional:
#   STUDENT_CHECKPOINT=/path/to/student.pt DISTILL_MODE=hybrid DEBUG_MAX_STEPS=10 bash run_opd_distill_llm.sh
#   DISTILL_MODE=opsd INIT_CHECKPOINT=/path/to/init.pt TRAIN_DATA=/path/to/opsd.data.list CV_DATA=/path/to/opsd.data.list bash run_opd_distill_llm.sh

CONDA_ACTIVATE=${CONDA_ACTIVATE:-/home/environment3/xmren/miniconda3/bin/activate}
CONDA_ENV=${CONDA_ENV:-/home/environment3/xmren/miniconda3/envs/cosyvoice/}
source "$CONDA_ACTIVATE" "$CONDA_ENV"

PRETRAINED_MODEL_DIR=${PRETRAINED_MODEL_DIR:-/home/work_nfs23/hkxie/code/CosyVoice/pretrained_models/CosyVoice2-0.5B}
TRAIN_DATA=${TRAIN_DATA:-/home/node62_data/hkxie/data/hw_data/train.data.list}
CV_DATA=${CV_DATA:-/home/node62_data/hkxie/data/hw_data/dev.data.list}
STUDENT_CHECKPOINT=${STUDENT_CHECKPOINT:-}
INIT_CHECKPOINT=${INIT_CHECKPOINT:-}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F "," '{print NF}')
JOB_ID=${JOB_ID:-1989}
RDZV_ENDPOINT=${RDZV_ENDPOINT:-localhost:1235}
DIST_BACKEND=${DIST_BACKEND:-nccl}
TRAIN_ENGINE=${TRAIN_ENGINE:-torch_ddp}
NUM_WORKERS=${NUM_WORKERS:-1}
PREFETCH=${PREFETCH:-200}
DISTILL_MODE=${DISTILL_MODE:-forced}
if [[ "$DISTILL_MODE" == "opsd" ]]; then
  CONFIG=${CONFIG:-conf/cosyvoice2_sft_1e-6_spk.yaml}
else
  CONFIG=${CONFIG:-conf/cosyvoice2_sft_1e-5_spk.yaml}
fi
TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT:-}
KD_TOP_K=${KD_TOP_K:-16}
KD_LOSS=${KD_LOSS:-reverse_kl_topk}
if [[ "$DISTILL_MODE" == "opsd" ]]; then
  KD_WEIGHT=${KD_WEIGHT:-1.0}
  EMA_TEACHER_WEIGHT=${EMA_TEACHER_WEIGHT:-0.0}
else
  KD_WEIGHT=${KD_WEIGHT:-0.2}
  EMA_TEACHER_WEIGHT=${EMA_TEACHER_WEIGHT:-0.05}
fi
EMA_DECAY=${EMA_DECAY:-0.999}
KD_TEMPERATURE=${KD_TEMPERATURE:-1.0}
ONLINE_START_STEP=${ONLINE_START_STEP:-2000}
ONLINE_INTERVAL=${ONLINE_INTERVAL:-4}
DEBUG_MAX_STEPS=${DEBUG_MAX_STEPS:-0}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-$([[ "$DISTILL_MODE" == "opsd" ]] && echo 500 || echo 0)}
MAX_EPOCH=${MAX_EPOCH:-}
SAVE_PER_STEP=${SAVE_PER_STEP:-$([[ "$DISTILL_MODE" == "opsd" ]] && echo 20 || true)}
LOG_INTERVAL=${LOG_INTERVAL:-$([[ "$DISTILL_MODE" == "opsd" ]] && echo 1 || true)}
TRAIN_BRANCH_MODE=${TRAIN_BRANCH_MODE:-$([[ "$DISTILL_MODE" == "opsd" ]] && echo unistream || echo auto)}
JOIN_TIMEOUT=${JOIN_TIMEOUT:-$([[ "$DISTILL_MODE" == "opsd" ]] && echo 1800 || echo 60)}
ACCUM_GRAD=${ACCUM_GRAD:-}
MAX_FRAMES_IN_BATCH=${MAX_FRAMES_IN_BATCH:-$([[ "$DISTILL_MODE" == "opsd" ]] && echo 8000 || true)}
SKIP_CV_ON_STEP_SAVE=${SKIP_CV_ON_STEP_SAVE:-$([[ "$DISTILL_MODE" == "opsd" ]] && echo 1 || echo 0)}
RUN_NAME=${RUN_NAME:-opd_distill_${DISTILL_MODE}_topk${KD_TOP_K}}
MODEL_DIR=${MODEL_DIR:-$(pwd)/exp/cosyvoice2/${RUN_NAME}/${TRAIN_ENGINE}}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-$(pwd)/tensorboard/cosyvoice2/${RUN_NAME}/${TRAIN_ENGINE}}
SUMMARIZE_METRICS=${SUMMARIZE_METRICS:-0}
METRIC_OUTPUT_DIR=${METRIC_OUTPUT_DIR:-$(pwd)/testout/opd_metric_summary/${RUN_NAME}}
VALIDATE_DATA=${VALIDATE_DATA:-1}
VALIDATE_MAX_RECORDS=${VALIDATE_MAX_RECORDS:-2000}
VALIDATE_MAX_FILES_PER_ENTRY=${VALIDATE_MAX_FILES_PER_ENTRY:-100}
VALIDATE_SPEECH_TOKEN_SIZE=${VALIDATE_SPEECH_TOKEN_SIZE:-6561}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_CKPT=${VALIDATE_CKPT:-$DRY_RUN}

if [[ -z "$MAX_EPOCH" && "$DISTILL_MODE" == "opsd" && "$MAX_TRAIN_STEPS" != "0" ]]; then
  MAX_EPOCH=1000000
fi

export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}

require_file() {
  local path="$1"
  local name="$2"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] ${name} not found: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  local name="$2"
  if [[ ! -d "$path" ]]; then
    echo "[ERROR] ${name} not found: ${path}" >&2
    exit 1
  fi
}

case "$DISTILL_MODE" in
  off|forced|online|hybrid|opsd) ;;
  *)
    echo "[ERROR] DISTILL_MODE must be off|forced|online|hybrid|opsd, got ${DISTILL_MODE}" >&2
    exit 1
    ;;
esac

require_file "$CONFIG" "CONFIG"
require_file "$TRAIN_DATA" "TRAIN_DATA"
require_file "$CV_DATA" "CV_DATA"
require_dir "$PRETRAINED_MODEL_DIR" "PRETRAINED_MODEL_DIR"
require_dir "$PRETRAINED_MODEL_DIR/CosyVoice-BlankEN" "Qwen pretrain dir"

if [[ -n "$INIT_CHECKPOINT" ]]; then
  require_file "$INIT_CHECKPOINT" "INIT_CHECKPOINT"
  STUDENT_CHECKPOINT=${STUDENT_CHECKPOINT:-$INIT_CHECKPOINT}
  TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT:-$INIT_CHECKPOINT}
fi

CHECKPOINT_ARGS=()
if [[ -n "$STUDENT_CHECKPOINT" ]]; then
  require_file "$STUDENT_CHECKPOINT" "STUDENT_CHECKPOINT"
  CHECKPOINT_ARGS=(--checkpoint "$STUDENT_CHECKPOINT")
fi

DISTILL_ARGS=(
  --distill_mode "$DISTILL_MODE"
  --kd_top_k "$KD_TOP_K"
  --kd_loss "$KD_LOSS"
  --kd_weight "$KD_WEIGHT"
  --ema_teacher_weight "$EMA_TEACHER_WEIGHT"
  --ema_decay "$EMA_DECAY"
  --kd_temperature "$KD_TEMPERATURE"
  --online_start_step "$ONLINE_START_STEP"
  --online_interval "$ONLINE_INTERVAL"
)

SAVE_ARGS=()
if [[ -n "$SAVE_PER_STEP" ]]; then
  if ! [[ "$SAVE_PER_STEP" =~ ^-?[0-9]+$ ]]; then
    echo "[ERROR] SAVE_PER_STEP must be an integer, got ${SAVE_PER_STEP}" >&2
    exit 1
  fi
  if (( SAVE_PER_STEP < -1 )); then
    echo "[ERROR] SAVE_PER_STEP must be >= -1, got ${SAVE_PER_STEP}" >&2
    exit 1
  fi
  SAVE_ARGS=(--save_per_step "$SAVE_PER_STEP")
fi

LOG_ARGS=()
if [[ -n "$LOG_INTERVAL" ]]; then
  if ! [[ "$LOG_INTERVAL" =~ ^[0-9]+$ ]] || (( LOG_INTERVAL <= 0 )); then
    echo "[ERROR] LOG_INTERVAL must be a positive integer, got ${LOG_INTERVAL}" >&2
    exit 1
  fi
  LOG_ARGS=(--log_interval "$LOG_INTERVAL")
fi

STEP_ARGS=()
if [[ -n "$MAX_TRAIN_STEPS" ]]; then
  if ! [[ "$MAX_TRAIN_STEPS" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] MAX_TRAIN_STEPS must be a non-negative integer, got ${MAX_TRAIN_STEPS}" >&2
    exit 1
  fi
  STEP_ARGS=(--max_train_steps "$MAX_TRAIN_STEPS")
fi

if ! [[ "$JOIN_TIMEOUT" =~ ^[0-9]+$ ]] || (( JOIN_TIMEOUT <= 0 )); then
  echo "[ERROR] JOIN_TIMEOUT must be a positive integer, got ${JOIN_TIMEOUT}" >&2
  exit 1
fi

EPOCH_ARGS=()
if [[ -n "$MAX_EPOCH" ]]; then
  if ! [[ "$MAX_EPOCH" =~ ^[0-9]+$ ]] || (( MAX_EPOCH <= 0 )); then
    echo "[ERROR] MAX_EPOCH must be a positive integer, got ${MAX_EPOCH}" >&2
    exit 1
  fi
  EPOCH_ARGS=(--max_epoch "$MAX_EPOCH")
fi

BATCH_ARGS=()
if [[ -n "$MAX_FRAMES_IN_BATCH" ]]; then
  if ! [[ "$MAX_FRAMES_IN_BATCH" =~ ^[0-9]+$ ]] || (( MAX_FRAMES_IN_BATCH <= 0 )); then
    echo "[ERROR] MAX_FRAMES_IN_BATCH must be a positive integer, got ${MAX_FRAMES_IN_BATCH}" >&2
    exit 1
  fi
  BATCH_ARGS=(--max_frames_in_batch "$MAX_FRAMES_IN_BATCH")
fi

ACCUM_ARGS=()
if [[ -n "$ACCUM_GRAD" ]]; then
  if ! [[ "$ACCUM_GRAD" =~ ^[0-9]+$ ]] || (( ACCUM_GRAD <= 0 )); then
    echo "[ERROR] ACCUM_GRAD must be a positive integer, got ${ACCUM_GRAD}" >&2
    exit 1
  fi
  ACCUM_ARGS=(--accum_grad "$ACCUM_GRAD")
fi

STEP_SAVE_ARGS=()
if [[ "$SKIP_CV_ON_STEP_SAVE" == "1" ]]; then
  STEP_SAVE_ARGS=(--skip_cv_on_step_save)
fi

if [[ "$DISTILL_MODE" != "off" ]]; then
  if [[ -z "$TEACHER_CHECKPOINT" ]]; then
    echo "Set TEACHER_CHECKPOINT=/path/to/teacher.pt when DISTILL_MODE=${DISTILL_MODE}" >&2
    exit 1
  fi
  require_file "$TEACHER_CHECKPOINT" "TEACHER_CHECKPOINT"
  DISTILL_ARGS+=(--teacher_checkpoint "$TEACHER_CHECKPOINT")
fi

mkdir -p "$MODEL_DIR" "$TENSORBOARD_DIR"
MANIFEST_PATH="$MODEL_DIR/opd_run_manifest.env"
{
  echo "RUN_NAME=${RUN_NAME}"
  echo "DATE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "HOSTNAME=$(hostname)"
  echo "PWD=$(pwd)"
  echo "CONFIG=${CONFIG}"
  echo "TRAIN_DATA=${TRAIN_DATA}"
  echo "CV_DATA=${CV_DATA}"
  echo "PRETRAINED_MODEL_DIR=${PRETRAINED_MODEL_DIR}"
  echo "INIT_CHECKPOINT=${INIT_CHECKPOINT}"
  echo "STUDENT_CHECKPOINT=${STUDENT_CHECKPOINT}"
  echo "TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT}"
  echo "DISTILL_MODE=${DISTILL_MODE}"
  echo "KD_TOP_K=${KD_TOP_K}"
  echo "KD_LOSS=${KD_LOSS}"
  echo "KD_WEIGHT=${KD_WEIGHT}"
  echo "EMA_TEACHER_WEIGHT=${EMA_TEACHER_WEIGHT}"
  echo "EMA_DECAY=${EMA_DECAY}"
  echo "KD_TEMPERATURE=${KD_TEMPERATURE}"
  echo "ONLINE_START_STEP=${ONLINE_START_STEP}"
  echo "ONLINE_INTERVAL=${ONLINE_INTERVAL}"
  echo "TRAIN_BRANCH_MODE=${TRAIN_BRANCH_MODE}"
  echo "ACCUM_GRAD=${ACCUM_GRAD}"
  echo "DEBUG_MAX_STEPS=${DEBUG_MAX_STEPS}"
  echo "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
  echo "MAX_EPOCH=${MAX_EPOCH}"
  echo "SAVE_PER_STEP=${SAVE_PER_STEP}"
  echo "LOG_INTERVAL=${LOG_INTERVAL}"
  echo "JOIN_TIMEOUT=${JOIN_TIMEOUT}"
  echo "SKIP_CV_ON_STEP_SAVE=${SKIP_CV_ON_STEP_SAVE}"
  echo "MAX_FRAMES_IN_BATCH=${MAX_FRAMES_IN_BATCH}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "NUM_GPUS=${NUM_GPUS}"
  echo "TRAIN_ENGINE=${TRAIN_ENGINE}"
  echo "MODEL_DIR=${MODEL_DIR}"
  echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}"
  echo "VALIDATE_DATA=${VALIDATE_DATA}"
  echo "VALIDATE_MAX_RECORDS=${VALIDATE_MAX_RECORDS}"
  echo "VALIDATE_MAX_FILES_PER_ENTRY=${VALIDATE_MAX_FILES_PER_ENTRY}"
  echo "VALIDATE_SPEECH_TOKEN_SIZE=${VALIDATE_SPEECH_TOKEN_SIZE}"
  echo "VALIDATE_CKPT=${VALIDATE_CKPT}"
  echo "DRY_RUN=${DRY_RUN}"
} > "$MANIFEST_PATH"

echo "Run CosyVoice2 LLM distillation: mode=${DISTILL_MODE}, teacher=${TEACHER_CHECKPOINT}"
echo "Manifest: ${MANIFEST_PATH}"

if [[ "$VALIDATE_DATA" == "1" ]]; then
  VALIDATE_DATA_ARGS=()
  if [[ "$DISTILL_MODE" == "opsd" ]]; then
    VALIDATE_DATA_ARGS+=(--require-prompt-fields)
  fi
  python3 validate_opd_data.py \
    --train-data "$TRAIN_DATA" \
    --cv-data "$CV_DATA" \
    --speech-token-size "$VALIDATE_SPEECH_TOKEN_SIZE" \
    --max-records-per-split "$VALIDATE_MAX_RECORDS" \
    --max-files-per-entry "$VALIDATE_MAX_FILES_PER_ENTRY" \
    "${VALIDATE_DATA_ARGS[@]}"
fi

if [[ "$VALIDATE_CKPT" == "1" ]]; then
  CHECKPOINT_VALIDATE_ARGS=(
    --config "$CONFIG"
    --qwen-pretrain-path "$PRETRAINED_MODEL_DIR/CosyVoice-BlankEN"
    --distill-mode "$DISTILL_MODE"
    --train-branch-mode "$TRAIN_BRANCH_MODE"
    --kd-weight "$KD_WEIGHT"
    --ema-teacher-weight "$EMA_TEACHER_WEIGHT"
  )
  if [[ -n "$STUDENT_CHECKPOINT" ]]; then
    CHECKPOINT_VALIDATE_ARGS+=(--student-checkpoint "$STUDENT_CHECKPOINT")
  fi
  if [[ -n "$TEACHER_CHECKPOINT" ]]; then
    CHECKPOINT_VALIDATE_ARGS+=(--teacher-checkpoint "$TEACHER_CHECKPOINT")
  fi
  python3 validate_opd_checkpoint.py "${CHECKPOINT_VALIDATE_ARGS[@]}"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1: preflight completed; skip torchrun."
  exit 0
fi

torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" \
  --rdzv_id="$JOB_ID" --rdzv_backend="c10d" --rdzv_endpoint="$RDZV_ENDPOINT" \
  ../../cosyvoice/bin/train.py \
  --train_engine "$TRAIN_ENGINE" \
  --config "$CONFIG" \
  --train_data "$TRAIN_DATA" \
  --cv_data "$CV_DATA" \
  --qwen_pretrain_path "$PRETRAINED_MODEL_DIR/CosyVoice-BlankEN" \
  --onnx_path "$PRETRAINED_MODEL_DIR" \
  --model llm \
  "${CHECKPOINT_ARGS[@]}" \
  --model_dir "$MODEL_DIR" \
  --tensorboard_dir "$TENSORBOARD_DIR" \
  --ddp.dist_backend "$DIST_BACKEND" \
  --num_workers "$NUM_WORKERS" \
  --prefetch "$PREFETCH" \
  --pin_memory \
  --use_amp \
  --train_branch_mode "$TRAIN_BRANCH_MODE" \
  "${DISTILL_ARGS[@]}" \
  --debug_max_steps "$DEBUG_MAX_STEPS" \
  "${STEP_ARGS[@]}" \
  "${EPOCH_ARGS[@]}" \
  "${SAVE_ARGS[@]}" \
  "${LOG_ARGS[@]}" \
  "${STEP_SAVE_ARGS[@]}" \
  "${BATCH_ARGS[@]}" \
  "${ACCUM_ARGS[@]}" \
  --timeout "$JOIN_TIMEOUT" \
  --deepspeed_config ./conf/ds_stage2.json \
  --deepspeed.save_states model+optimizer

if [[ "$SUMMARIZE_METRICS" == "1" ]]; then
  METRIC_REQUIRE_ARGS=()
  if [[ "$DISTILL_MODE" == "forced" || "$DISTILL_MODE" == "hybrid" ]]; then
    METRIC_REQUIRE_ARGS+=(--require TRAIN/external_kd_loss --require TRAIN/external_kd_topk_overlap)
  fi
  if [[ "$DISTILL_MODE" == "online" || "$DISTILL_MODE" == "hybrid" ]]; then
    METRIC_REQUIRE_ARGS+=(--require TRAIN/online_external_kd_loss --require TRAIN/online_external_kd_topk_overlap --require TRAIN/online_sample_token_count)
  fi
  if [[ "$DISTILL_MODE" == "opsd" ]]; then
    METRIC_REQUIRE_ARGS+=(--require TRAIN/opsd_kd_loss --require TRAIN/opsd_kd_topk_overlap --require TRAIN/online_sample_token_count)
  fi
  python3 summarize_opd_tensorboard.py \
    --input "$TENSORBOARD_DIR" \
    --output_dir "$METRIC_OUTPUT_DIR" \
    --curves \
    "${METRIC_REQUIRE_ARGS[@]}"
fi
