#!/usr/bin/env python3
# Copyright (c) 2026
#
# CPU checkpoint/config preflight for CosyVoice2 OPD distillation.

import argparse
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "Matcha-TTS"))

from hyperpyyaml import load_hyperpyyaml

from cosyvoice.utils.distill_utils import (  # noqa: E402
    _torch_load_weights,
    extract_model_state_dict,
    load_checkpoint_to_model,
)


def build_llm(config_path: Path, qwen_pretrain_path: str, train_branch_mode: str):
    override_dict = {
        "flow": None,
        "hift": None,
        "hifigan": None,
        "mel_spec_transform1": None,
        "feat_extractor": None,
        "data_pipeline_gan": [],
        "qwen_pretrain_path": qwen_pretrain_path,
    }
    if train_branch_mode:
        override_dict["train_branch_mode"] = train_branch_mode
    with config_path.open("r", encoding="utf-8") as fin:
        configs = load_hyperpyyaml(fin, overrides=override_dict)
    return configs["llm"]


def checkpoint_summary(checkpoint: Path) -> str:
    checkpoint_obj = _torch_load_weights(str(checkpoint))
    state_dict = extract_model_state_dict(checkpoint_obj, str(checkpoint))
    epoch = checkpoint_obj.get("epoch", "NA") if isinstance(checkpoint_obj, dict) else "NA"
    step = checkpoint_obj.get("step", "NA") if isinstance(checkpoint_obj, dict) else "NA"
    first_key = next(iter(state_dict))
    return "tensor_keys={} epoch={} step={} first_key={}".format(
        len(state_dict), epoch, step, first_key)


def validate_one(label: str, model, checkpoint: str) -> None:
    if not checkpoint:
        print("{}=none".format(label))
        return
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError("{} not found: {}".format(label, path))
    print("{}={}".format(label, path))
    print("{}_summary={}".format(label, checkpoint_summary(path)))
    missing_keys, unexpected_keys = load_checkpoint_to_model(
        model,
        str(path),
        strict_match=True,
        label=label,
    )
    print("{}_missing_keys={}".format(label, len(missing_keys)))
    print("{}_unexpected_keys={}".format(label, len(unexpected_keys)))


def parse_args():
    parser = argparse.ArgumentParser(description="Validate OPD checkpoint/config compatibility on CPU")
    parser.add_argument("--config", required=True)
    parser.add_argument("--qwen-pretrain-path", required=True)
    parser.add_argument("--student-checkpoint", default="")
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--distill-mode", default="forced", choices=["off", "forced", "online", "hybrid", "opsd"])
    parser.add_argument("--train-branch-mode", default="auto", choices=["auto", "unistream", "bistream"])
    parser.add_argument("--kd-weight", type=float, default=0.2)
    parser.add_argument("--ema-teacher-weight", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.distill_mode != "off":
        if args.kd_weight < 0 or args.ema_teacher_weight < 0:
            raise ValueError("kd weights must be non-negative")
        if args.kd_weight == 0 and args.ema_teacher_weight == 0:
            raise ValueError("at least one teacher weight must be > 0")
        if args.distill_mode == "opsd" and (args.kd_weight <= 0 or args.ema_teacher_weight != 0):
            raise ValueError("distill-mode opsd requires kd-weight > 0 and ema-teacher-weight 0")
        if not args.teacher_checkpoint:
            raise ValueError("--teacher-checkpoint is required when distill-mode != off")
    model = build_llm(Path(args.config), args.qwen_pretrain_path, args.train_branch_mode)
    print("llm_class={}".format(model.__class__.__name__))
    if hasattr(model, "branch_mode"):
        print("llm_branch_mode={}".format(model.branch_mode))
    if hasattr(model, "speech_token_size"):
        print("speech_token_size={}".format(model.speech_token_size))
    validate_one("student_checkpoint", model, args.student_checkpoint)
    if args.distill_mode != "off":
        validate_one("teacher_checkpoint", model, args.teacher_checkpoint)
    print("opd_checkpoint_validation_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
