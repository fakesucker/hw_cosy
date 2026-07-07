#!/usr/bin/env python3
"""Lightweight unit checks for CosyVoice2 OPD distillation utilities."""

from __future__ import annotations

import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cosyvoice.utils.distill_utils import (  # noqa: E402
    DistillManager,
    TeacherManager,
    extract_model_state_dict,
    load_checkpoint_to_model,
    load_teacher_checkpoint,
    sparse_forward_kl_topk,
    sparse_reverse_kl_topk,
    sparse_topk_kd_loss,
)


def test_sparse_kd_wrappers() -> None:
    student = torch.randn(2, 5, 11, requires_grad=True)
    teacher = torch.randn(2, 5, 11, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 1, 0], [1, 0, 1, 1, 1]], dtype=torch.bool)

    reverse_loss, reverse_metrics = sparse_reverse_kl_topk(student, teacher, mask, top_k=4)
    reverse_ref, reverse_ref_metrics = sparse_topk_kd_loss(
        student, teacher, mask, top_k=4, loss_type="reverse_kl_topk"
    )
    forward_loss, forward_metrics = sparse_forward_kl_topk(student, teacher, mask, top_k=4)
    forward_ref, forward_ref_metrics = sparse_topk_kd_loss(
        student, teacher, mask, top_k=4, loss_type="forward_kl_topk"
    )

    assert reverse_loss.ndim == 0 and torch.isfinite(reverse_loss)
    assert forward_loss.ndim == 0 and torch.isfinite(forward_loss)
    assert torch.allclose(reverse_loss, reverse_ref)
    assert torch.allclose(forward_loss, forward_ref)
    expected_metrics = ["kd_token_count", "kd_top1_agree", "kd_topk_overlap"]
    assert sorted(reverse_metrics) == sorted(reverse_ref_metrics) == expected_metrics
    assert sorted(forward_metrics) == sorted(forward_ref_metrics) == expected_metrics

    (reverse_loss + forward_loss).backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert torch.allclose(student.grad[~mask], torch.zeros_like(student.grad[~mask]))
    assert teacher.grad is None


def test_sparse_kd_zero_mask_metrics() -> None:
    student = torch.randn(1, 3, 7, requires_grad=True)
    teacher = torch.randn(1, 3, 7)
    mask = torch.zeros(1, 3, dtype=torch.bool)
    loss, metrics = sparse_topk_kd_loss(student, teacher, mask, top_k=3)
    assert torch.isclose(loss.float(), torch.tensor(0.0))
    assert sorted(metrics) == ["kd_token_count", "kd_top1_agree", "kd_topk_overlap"]
    assert all(torch.isclose(value.float(), torch.tensor(0.0)) for value in metrics.values())
    loss.backward()
    assert student.grad is not None
    assert torch.allclose(student.grad, torch.zeros_like(student.grad))


def test_ema_and_teacher_manager() -> None:
    student_model = nn.Linear(3, 2, bias=True)
    ema_model = nn.Linear(3, 2, bias=True)
    ema_model.load_state_dict(student_model.state_dict())
    manager = TeacherManager(
        external_teacher=None,
        ema_teacher=ema_model,
        mode="forced",
        kd_top_k=4,
        kd_loss="reverse_kl_topk",
        kd_weight=0.0,
        ema_teacher_weight=0.1,
        ema_decay=0.5,
        kd_temperature=1.0,
        online_start_step=0,
        online_interval=4,
    )

    old_weight = ema_model.weight.detach().clone()
    old_bias = ema_model.bias.detach().clone()
    with torch.no_grad():
        student_model.weight.add_(2.0)
        student_model.bias.sub_(1.0)
    expected_weight = old_weight * 0.5 + student_model.weight.detach() * 0.5
    expected_bias = old_bias * 0.5 + student_model.bias.detach() * 0.5
    manager.update_ema(student_model)

    assert torch.allclose(ema_model.weight, expected_weight)
    assert torch.allclose(ema_model.bias, expected_bias)
    assert all(not param.requires_grad for param in ema_model.parameters())


class Qwen2LM(nn.Module):
    def __init__(self, logit_bias: float = 0.0, sample_tokens=(1, 2)) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.tensor(1.0))
        self.logit_bias = nn.Parameter(torch.tensor(logit_bias))
        self.speech_token_size = 8
        self.sample_tokens = sample_tokens
        self.prompt_conditioned_calls = 0

    def forward_logits(self, batch, device, branch_choices=None, speech_token_override=None, prompt_conditioned=False):
        batch_size = batch["text_token"].size(0)
        speech_len = 2 if speech_token_override is None else int(speech_token_override[1][0].item())
        prompt_len = 0
        if prompt_conditioned:
            self.prompt_conditioned_calls += 1
            prompt_len = int(batch["prompt_speech_token_len"][0].item())
        seq_len = speech_len + prompt_len + 1
        base = torch.arange(10, dtype=self.param.dtype, device=self.param.device).reshape(1, 1, 10)
        logits = (self.param + base * self.logit_bias).expand(batch_size, seq_len, 10)
        lm_target = torch.zeros(batch_size, seq_len, dtype=torch.long)
        valid_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        speech_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        if speech_len > 0:
            speech_mask[:, seq_len - speech_len - 1:seq_len - 1] = True
        return {
            "loss": self.param * 0.0 + 7.0,
            "loss_ce": self.param * 0.0 + 7.0,
            "acc": self.param * 0.0,
            "logits": logits,
            "lm_target": lm_target,
            "valid_mask": valid_mask,
            "speech_mask": speech_mask,
            "speech_ce_loss": self.param.detach() * 0.0 + 7.0,
            "branch_choices": [False] * batch_size,
        }

    def inference(self, *args, **kwargs):
        for token in self.sample_tokens:
            yield token


def test_online_mode_does_not_train_ce() -> None:
    model = Qwen2LM()
    batch = {
        "utts": ["utt"],
        "text_token": torch.ones(1, 2, dtype=torch.long),
        "text_token_len": torch.tensor([2], dtype=torch.int32),
        "speech_token": torch.ones(1, 2, dtype=torch.long),
        "speech_token_len": torch.tensor([2], dtype=torch.int32),
    }
    teacher = deepcopy(model)
    manager = DistillManager(
        external_teacher=teacher,
        ema_teacher=None,
        mode="online",
        kd_top_k=4,
        kd_loss="reverse_kl_topk",
        kd_weight=0.1,
        ema_teacher_weight=0.0,
        ema_decay=0.5,
        kd_temperature=1.0,
        online_start_step=0,
        online_interval=1,
    )
    loss_dict = manager.forward(model, batch, torch.device("cpu"), {"step": 123})
    assert torch.isclose(loss_dict["ce_loss"].float(), torch.tensor(7.0))
    assert torch.isclose(loss_dict["loss"].float(), torch.tensor(0.0))
    assert torch.isclose(loss_dict["kd_loss"].float(), torch.tensor(0.0))
    assert torch.isclose(loss_dict["online_sample_token_count"].float(), torch.tensor(2.0))
    assert torch.isclose(loss_dict["online_external_kd_token_count"].float(), torch.tensor(2.0))


def test_online_no_sample_has_stable_zero_metrics() -> None:
    model = Qwen2LM(sample_tokens=(99,))
    teacher = deepcopy(model)
    batch = {
        "utts": ["utt"],
        "text_token": torch.ones(1, 2, dtype=torch.long),
        "text_token_len": torch.tensor([2], dtype=torch.int32),
        "speech_token": torch.ones(1, 2, dtype=torch.long),
        "speech_token_len": torch.tensor([2], dtype=torch.int32),
    }
    manager = DistillManager(
        external_teacher=teacher,
        ema_teacher=None,
        mode="online",
        kd_top_k=4,
        kd_loss="reverse_kl_topk",
        kd_weight=0.1,
        ema_teacher_weight=0.0,
        ema_decay=0.5,
        kd_temperature=1.0,
        online_start_step=0,
        online_interval=1,
    )
    loss_dict = manager.forward(model, batch, torch.device("cpu"), {"step": 123})
    for key in [
        "online_sample_token_count",
        "online_sample_batch_size",
        "online_external_kd_loss",
        "online_external_speech_kd_loss",
        "online_ema_kd_loss",
        "online_ema_speech_kd_loss",
    ]:
        assert key in loss_dict
        assert torch.isclose(loss_dict[key].float(), torch.tensor(0.0))


def test_forced_kd_adds_teacher_signal() -> None:
    model = Qwen2LM(logit_bias=0.0)
    teacher = Qwen2LM(logit_bias=0.25)
    batch = {
        "utts": ["utt"],
        "text_token": torch.ones(1, 2, dtype=torch.long),
        "text_token_len": torch.tensor([2], dtype=torch.int32),
        "speech_token": torch.ones(1, 2, dtype=torch.long),
        "speech_token_len": torch.tensor([2], dtype=torch.int32),
    }
    manager = DistillManager(
        external_teacher=teacher,
        ema_teacher=None,
        mode="forced",
        kd_top_k=4,
        kd_loss="forward_kl_topk",
        kd_weight=0.5,
        ema_teacher_weight=0.0,
        ema_decay=0.5,
        kd_temperature=1.0,
        online_start_step=0,
        online_interval=1,
    )
    loss_dict = manager.forward(model, batch, torch.device("cpu"), {"step": 0})
    assert loss_dict["external_kd_loss"].item() > 0
    expected = loss_dict["ce_loss"] + loss_dict["external_kd_loss"] * 0.5
    assert torch.allclose(loss_dict["loss"], expected)
    assert "external_kd_topk_overlap" in loss_dict


def test_opsd_uses_prompt_conditioned_teacher_without_ce() -> None:
    model = Qwen2LM(logit_bias=0.0, sample_tokens=(1, 2))
    teacher = Qwen2LM(logit_bias=0.25, sample_tokens=(1, 2))
    batch = {
        "utts": ["utt"],
        "text_token": torch.ones(1, 2, dtype=torch.long),
        "text_token_len": torch.tensor([2], dtype=torch.int32),
        "speech_token": torch.zeros(1, 4, dtype=torch.long),
        "speech_token_len": torch.tensor([4], dtype=torch.int32),
        "prompt_text_token": torch.ones(1, 3, dtype=torch.long),
        "prompt_text_token_len": torch.tensor([3], dtype=torch.int32),
        "prompt_speech_token": torch.ones(1, 5, dtype=torch.long),
        "prompt_speech_token_len": torch.tensor([5], dtype=torch.int32),
    }
    manager = DistillManager(
        external_teacher=teacher,
        ema_teacher=None,
        mode="opsd",
        kd_top_k=4,
        kd_loss="forward_kl_topk",
        kd_weight=0.5,
        ema_teacher_weight=0.0,
        ema_decay=0.5,
        kd_temperature=1.0,
        online_start_step=0,
        online_interval=1,
    )
    loss_dict = manager.forward(model, batch, torch.device("cpu"), {"step": 0})
    assert teacher.prompt_conditioned_calls == 1
    assert torch.isclose(loss_dict["online_sample_token_count"].float(), torch.tensor(2.0))
    assert loss_dict["opsd_kd_loss"].item() > 0
    assert torch.allclose(loss_dict["loss"], loss_dict["opsd_kd_loss"] * 0.5)
    assert torch.isclose(loss_dict["ce_loss"].float(), torch.tensor(7.0))


def test_online_batch_clones_inference_tensors() -> None:
    model = Qwen2LM(logit_bias=0.0, sample_tokens=(1, 2))
    with torch.inference_mode():
        batch = {
            "utts": ["utt"],
            "text_token": torch.ones(1, 2, dtype=torch.long),
            "text_token_len": torch.tensor([2], dtype=torch.int32),
            "speech_token": torch.zeros(1, 4, dtype=torch.long),
            "speech_token_len": torch.tensor([4], dtype=torch.int32),
            "prompt_text_token": torch.ones(1, 3, dtype=torch.long),
            "prompt_text_token_len": torch.tensor([3], dtype=torch.int32),
            "prompt_speech_token": torch.ones(1, 5, dtype=torch.long),
            "prompt_speech_token_len": torch.tensor([5], dtype=torch.int32),
        }
    assert batch["text_token"].is_inference()
    manager = DistillManager(
        external_teacher=deepcopy(model),
        ema_teacher=None,
        mode="opsd",
        kd_top_k=4,
        kd_loss="forward_kl_topk",
        kd_weight=0.5,
        ema_teacher_weight=0.0,
        ema_decay=0.5,
        kd_temperature=1.0,
        online_start_step=0,
        online_interval=1,
    )
    online_batch = manager._sample_online_batch(model, batch, torch.device("cpu"), step=0)
    assert online_batch is not None
    for value in online_batch.values():
        if torch.is_tensor(value):
            assert not value.is_inference()


def test_teacher_checkpoint_strict_match() -> None:
    model = nn.Linear(2, 2)
    with tempfile.NamedTemporaryFile(suffix=".pt") as fout:
        torch.save({"weight": torch.ones(2, 2)}, fout.name)
        try:
            load_teacher_checkpoint(model, fout.name)
        except ValueError as exc:
            assert "does not match model exactly" in str(exc)
        else:
            raise AssertionError("strict teacher checkpoint loading should reject missing keys")


def test_checkpoint_extraction_and_strict_loader() -> None:
    model = nn.Linear(2, 2)
    checkpoint = {
        "state_dict": {
            "module.weight": torch.ones(2, 2),
            "module.bias": torch.zeros(2),
        },
        "epoch": 3,
        "step": 12,
    }
    state_dict = extract_model_state_dict(checkpoint, "nested")
    assert sorted(state_dict) == ["bias", "weight"]
    with tempfile.NamedTemporaryFile(suffix=".pt") as fout:
        torch.save(checkpoint, fout.name)
        missing, unexpected = load_checkpoint_to_model(
            model,
            fout.name,
            strict_match=True,
            label="unit checkpoint",
        )
    assert missing == []
    assert unexpected == []
    assert torch.allclose(model.weight, torch.ones_like(model.weight))
    assert torch.allclose(model.bias, torch.zeros_like(model.bias))


def test_distill_requires_teacher_signal() -> None:
    try:
        DistillManager(
            external_teacher=None,
            ema_teacher=None,
            mode="forced",
            kd_top_k=4,
            kd_loss="reverse_kl_topk",
            kd_weight=0.0,
            ema_teacher_weight=0.0,
            ema_decay=0.5,
            kd_temperature=1.0,
            online_start_step=0,
            online_interval=1,
        )
    except ValueError as exc:
        assert "teacher weight" in str(exc)
    else:
        raise AssertionError("distill mode with zero teacher weights should fail")


def test_distill_rejects_invalid_online_schedule() -> None:
    for kwargs in [
        {"online_start_step": -1, "online_interval": 1},
        {"online_start_step": 0, "online_interval": 0},
    ]:
        try:
            DistillManager(
                external_teacher=Qwen2LM(),
                ema_teacher=None,
                mode="online",
                kd_top_k=4,
                kd_loss="reverse_kl_topk",
                kd_weight=0.1,
                ema_teacher_weight=0.0,
                ema_decay=0.5,
                kd_temperature=1.0,
                **kwargs,
            )
        except ValueError as exc:
            assert "online_" in str(exc)
        else:
            raise AssertionError("invalid online schedule should fail")


def main() -> None:
    test_sparse_kd_wrappers()
    test_sparse_kd_zero_mask_metrics()
    test_ema_and_teacher_manager()
    test_online_mode_does_not_train_ce()
    test_online_no_sample_has_stable_zero_metrics()
    test_forced_kd_adds_teacher_signal()
    test_opsd_uses_prompt_conditioned_teacher_without_ce()
    test_online_batch_clones_inference_tensors()
    test_teacher_checkpoint_strict_match()
    test_checkpoint_extraction_and_strict_loader()
    test_distill_requires_teacher_signal()
    test_distill_rejects_invalid_online_schedule()
    print("opd_distill_unit_ok")


if __name__ == "__main__":
    main()
