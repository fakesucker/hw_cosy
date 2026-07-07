# CosyVoice2 OPD Distillation

This note records the local OPD-style distillation entry points for CosyVoice2
LLM training. The implementation stays inside the native CosyVoice trainer and
does not migrate training to verl.

## Training

Use `run_opd_distill_llm.sh` from this directory:

```bash
cd /home/work_nfs23/hkxie/hw_proj/CosyVoice/examples/huawei_sft

TEACHER_CHECKPOINT=/path/to/better_teacher.pt \
STUDENT_CHECKPOINT=/path/to/student_init.pt \
DISTILL_MODE=forced \
SUMMARIZE_METRICS=1 \
RUN_NAME=opd_forced_topk16 \
bash run_opd_distill_llm.sh
```

Important options:

- `DISTILL_MODE=off|forced|online|hybrid`
- `TEACHER_CHECKPOINT=/path/to/teacher.pt`, required unless `DISTILL_MODE=off`
- `KD_TOP_K=16`
- `KD_LOSS=reverse_kl_topk|forward_kl_topk`
- `KD_WEIGHT=0.2`
- `EMA_TEACHER_WEIGHT=0.05`
- `EMA_DECAY=0.999`
- `KD_TEMPERATURE=1.0`
- `ONLINE_START_STEP=2000`
- `ONLINE_INTERVAL=4`
- `DEBUG_MAX_STEPS=1` for a short smoke run
- `SAVE_PER_STEP=100` to save step checkpoints such as
  `epoch_0_step_40100.pt`; leave unset to use the YAML `save_per_step`
- `SUMMARIZE_METRICS=1` to export TensorBoard OPD metrics after a successful
  run
- `MODEL_DIR=/path/to/save/torch_ddp` and `TENSORBOARD_DIR=/path/to/log` to
  override the default output locations

When `DISTILL_MODE != off`, at least one of `KD_WEIGHT` or
`EMA_TEACHER_WEIGHT` must be greater than zero. This prevents accidentally
running a long job with distillation enabled but no teacher signal.
The external teacher checkpoint is loaded with strict model-key matching. When
`STUDENT_CHECKPOINT` is provided under distillation, it is also checked strictly
against the configured LLM model. A checkpoint with missing or unexpected model
parameters fails before training instead of being partially loaded silently.
`ONLINE_START_STEP` must be non-negative and `ONLINE_INTERVAL` must be greater
than zero. They control when the extra online branch runs in `hybrid` mode;
pure `online` mode performs online KD every training step.

Mode semantics:

- `forced`: train with CE on ground-truth speech tokens plus external-teacher
  KD and EMA-teacher KD on the same forced-token states.
- `online`: sample speech tokens from the current student and train only from
  teacher KD on sampled speech-token positions. EOS/fill targets are excluded
  from the online KD mask. Ground-truth CE is logged but not added to the
  training loss. Pure `online` mode samples every training step.
- `hybrid`: always applies forced-token CE/KD and additionally runs online KD
  after `ONLINE_START_STEP` every `ONLINE_INTERVAL` optimizer steps.
- `off`: normal SFT path without teacher checkpoint or KD logging.

The first implementation intentionally supports `--model llm` with
`torch_ddp`. It fail-fasts for `deepspeed + distill` and `dpo + distill`.
The launcher writes `opd_run_manifest.env` under `MODEL_DIR` before starting
training so each real run records teacher/student checkpoints, KD weights,
online schedule, optional step-save cadence, data lists, and device layout.

For OPD experiments, short checkpoint cadence is usually more useful than
epoch-only saving because the student can move toward the teacher within a few
hundred optimizer steps. A practical first sweep is `SAVE_PER_STEP=100` and
then infer `epoch_*_step_*.pt` at 100, 200, 500, and 1000 optimizer-step
increments before spending time on longer runs. Use `SAVE_PER_STEP=50` only
when the evaluation set is small enough or when debugging early-step behavior,
because each saved checkpoint is roughly the full LLM checkpoint size.

## Data

OPD distillation reuses the existing CosyVoice2 LLM SFT data. It does not need
pre-generated teacher audio, teacher logits, ASR rewards, mel features, or
speaker embeddings for the default `cosyvoice2_sft_1e-5_spk.yaml` path.

`TRAIN_DATA` and `CV_DATA` are list files. Each non-empty line points to a JSONL
file. Each JSONL record must provide:

```json
{"key": "utt_id", "txt": "text", "code": [12, 345, 678]}
```

The equivalent field names `utt`, `text`, and `speech_token` are also accepted.
`duration` is optional. `code` / `speech_token` may be a JSON list or a string
literal that parses to a list of integer speech-token ids. For the current
CosyVoice2 config, token ids must be in `[0, 6561)`.

Mode-specific use of the same data:

- `forced`: uses ground-truth `speech_token` for CE and teacher KD.
- `online`: samples new speech tokens from the student, but still uses
  `speech_token_len` from the original record to bound rollout length and log
  CE metrics.
- `hybrid`: uses the same SFT records for forced KD plus scheduled online KD.

By default the launcher validates a small prefix of train and CV data before
starting torchrun:

```bash
python3 validate_opd_data.py \
  --train-data /home/node62_data/hkxie/data/hw_data/train.data.list \
  --cv-data /home/node62_data/hkxie/data/hw_data/dev.data.list
```

Use `VALIDATE_DATA=0` to skip the preflight, or increase coverage with
`VALIDATE_MAX_RECORDS=0` to scan all listed records. If a list entry is a
directory, the validator scans JSONL-like files under it and caps each directory
with `VALIDATE_MAX_FILES_PER_ENTRY=100` by default.

Use `DRY_RUN=1` to run path checks, write `opd_run_manifest.env`, validate data,
validate checkpoint/config compatibility on CPU, and then exit before
`torchrun`. This is the recommended first check after setting real
`TEACHER_CHECKPOINT`, `STUDENT_CHECKPOINT`, `TRAIN_DATA`, and `CV_DATA`.
`VALIDATE_CKPT` defaults to the same value as `DRY_RUN`; set
`VALIDATE_CKPT=1` to run the CPU checkpoint preflight before a real training
launch as well.

## Smoke Checks

The trainer has been smoke-tested with temporary JSONL data for:

- `DISTILL_MODE=forced` on 1 GPU
- `DISTILL_MODE=hybrid` on 1 GPU with online KD active
- `DISTILL_MODE=online` on 1 GPU
- `DISTILL_MODE=forced` on 2 GPUs with DDP
- `DISTILL_MODE=off` on 1 GPU

Run the lightweight CPU unit checks with:

```bash
python3 examples/huawei_sft/test_opd_distill_unit.py
```

For a fast local check, use:

```bash
CUDA_VISIBLE_DEVICES=0 \
DEBUG_MAX_STEPS=1 \
DISTILL_MODE=forced \
TEACHER_CHECKPOINT=/path/to/teacher.pt \
STUDENT_CHECKPOINT=/path/to/student_init.pt \
bash run_opd_distill_llm.sh
```

Expected training logs include:

- `ce_loss`
- `speech_ce_loss`
- `external_kd_loss`
- `external_speech_kd_loss`
- `ema_kd_loss`
- `ema_speech_kd_loss`
- `kd_loss`
- `*_kd_topk_overlap` and `*_kd_top1_agree`

For `online` and `hybrid` with online KD active, logs also include
`online_external_*`, `online_ema_*`, and `online_sample_token_count`.
If a scheduled online step samples no speech tokens, the online KD metrics are
logged as zero-valued fields instead of disappearing from the log stream.

## Metric Summary

Use a unique `RUN_NAME` for each real run so TensorBoard event files do not mix
unrelated experiments. After training, export OPD curves and summaries with:

```bash
python3 examples/huawei_sft/summarize_opd_tensorboard.py \
  --input examples/huawei_sft/tensorboard/cosyvoice2/opd_forced_topk16/torch_ddp \
  --output_dir testout/opd_metric_summary/opd_forced_topk16 \
  --curves \
  --require TRAIN/external_kd_loss \
  --require TRAIN/external_kd_topk_overlap
```

For `hybrid` or `online` runs, also require online tags:

```bash
python3 examples/huawei_sft/summarize_opd_tensorboard.py \
  --input examples/huawei_sft/tensorboard/cosyvoice2/opd_hybrid_topk16/torch_ddp \
  --output_dir testout/opd_metric_summary/opd_hybrid_topk16 \
  --curves \
  --require TRAIN/online_external_kd_loss \
  --require TRAIN/online_external_kd_topk_overlap \
  --require TRAIN/online_sample_token_count
```

Outputs:

- `summary.tsv`: one row per scalar tag with count, first/last value, min/max,
  mean, delta, and slope per step
- `curves.tsv`: per-step scalar rows for plotting CE, KD loss, top-k overlap,
  token counts, and gradient/lr curves

By default the exporter deduplicates repeated scalar points at the same global
step and keeps the newest event value. Use `--dedupe_steps none` if you need the
raw event stream from a directory that was reused across restarts.

## Midterm Evaluation

After a real OPD training run produces `epoch_*_whole.pt`, compare baseline and
OPD checkpoints on `testset_midterm` with:

```bash
cd /home/work_nfs23/hkxie/hw_proj

OPD_MODEL_DIRS=/path/to/opd/torch_ddp \
BASELINE_MODEL_DIRS=/path/to/baseline/torch_ddp \
TOP_N=1 \
RUN_MODE=serial \
GPU_LIST=0 \
CosyVoice/examples/huawei_sft/run_opd_midterm_eval.sh
```

The wrapper reuses existing inference and CER scripts. It covers:

- set1: `testset_midterm/wer/ceping_cer_wer.lst`
- set2: `testset_midterm/wer/ceping_cer_wer_set2.lst`
- set3: `testset_midterm/wer/ceping_cer_wer_set3.lst`

Outputs:

- `testout/opd_midterm_eval/preflight_model_dirs.tsv`
- `testout/opd_midterm_eval/staged_model_dirs.tsv`
- `testout/opd_midterm_eval/summary_latency.tsv`
- `testout/opd_midterm_eval/summary_all.tsv`
- `testout/opd_midterm_eval/summary_cer_matrix.tsv`
- `testout/opd_midterm_eval/summary_compare.tsv`
- per-epoch `latency.tsv` under each inference output directory

`summary_compare.tsv` pairs `baseline_*` rows with `opd_*` rows within the
same set/meta and reports OPD-minus-baseline CER, RTF, utterance completion
latency, and first-audio latency deltas. Negative CER delta means OPD improved
CER.

Set `RUN_INFER=0` to reuse existing wav outputs and only run CER. Set
`RUN_CER=0` to only run inference. `VALIDATE_EVAL_INPUTS=1` is enabled by
default and fails fast if any requested OPD/baseline model directory is missing
or has no `epoch_*_whole.pt` checkpoint selected by `TOP_N`.
`STAGE_EVAL_MODEL_DIRS=1` is also enabled by default. It creates per-run
symlink directories such as `baseline_<run>` and `opd_<run>` before invoking the
legacy inference sweep so baseline and OPD outputs do not collide when both
input directories are named `torch_ddp`. If multiple inputs resolve to the same
alias, the later staged directories receive `_2`, `_3`, and so on.
For non-dry-run CER evaluation, `REQUIRE_CER_ROWS=1`, `REQUIRE_CER_OK=1`, and
`REQUIRE_COMPARE_ROWS=1` are enabled by default so empty CER summaries, failed
WER rows, or missing baseline-vs-OPD comparison rows fail the wrapper instead
of looking like a successful evaluation.

## Current Verification

Local checks completed for the implementation:

- Python compile check for `train.py`, `llm.py`, `executor.py`,
  `train_utils.py`, `distill_utils.py`, and `test_opd_distill_unit.py`
- CPU unit checks for sparse KL, mask gradients, teacher no-grad, EMA, online
  no-CE behavior, forced KD loss composition, zero-mask metrics, strict teacher
  checkpoint matching, zero-teacher-signal fail-fast, invalid online schedule
  rejection, and stable zero online metrics when no speech token is sampled
- shell syntax checks for `run_opd_distill_llm.sh` and
  `run_opd_midterm_eval.sh`
- TensorBoard metric export dry-run for hybrid smoke OPD tags
- launcher manifest and `SUMMARIZE_METRICS=1` auto metric export smoke
- 1-GPU forced smoke with KD metrics and top-k overlap logging
- 1-GPU hybrid smoke with online KD on sampled speech-token positions and
  online top-k overlap logging
- 1-GPU pure online smoke with CE logged only, training loss from online KD,
  and online KD token count matching sampled speech-token count
- 1-GPU pure online smoke with default `ONLINE_INTERVAL=4`, verifying online
  KD still runs every step
- 1-GPU `DISTILL_MODE=off` smoke with no teacher checkpoint and no KD metrics
- 2-GPU forced DDP smoke with per-rank KD metrics and no deadlock
- `run_opd_midterm_eval.sh` dry-run for set1/set2/set3 using a smoke
  `epoch_1_whole.pt`

Remaining acceptance work:

- launch a real OPD training run with a better external teacher checkpoint
- evaluate the resulting `epoch_*_whole.pt` against baseline on
  `testset_midterm` set1/set2/set3 and compare CER plus latency
