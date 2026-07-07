# Copyright (c) 2026
#
# Utilities for OPD-style sparse top-k distillation on CosyVoice LLM training.

import logging
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, 'module') else model


def _module_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    state_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    if len(state_dict) > 0 and all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict


def _torch_load_weights(checkpoint: str):
    try:
        return torch.load(checkpoint, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(checkpoint, map_location='cpu')


def extract_model_state_dict(checkpoint_dict, checkpoint: str = 'checkpoint') -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint_dict, dict):
        raise ValueError('{} is not a state_dict-like dict'.format(checkpoint))
    for key in ['state_dict', 'model_state_dict', 'model']:
        nested = checkpoint_dict.get(key)
        if isinstance(nested, dict):
            checkpoint_dict = nested
            break
    state_dict = _module_state_dict(checkpoint_dict)
    if len(state_dict) == 0:
        raise ValueError('{} does not contain tensor model parameters'.format(checkpoint))
    return state_dict


def load_checkpoint_to_model(model: torch.nn.Module,
                             checkpoint: str,
                             strict_match: bool = False,
                             label: str = 'checkpoint') -> Tuple[list, list]:
    checkpoint_dict = _torch_load_weights(checkpoint)
    state_dict = extract_model_state_dict(checkpoint_dict, checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    logging.info(
        'Loaded %s %s, tensor_keys=%d missing_keys=%d unexpected_keys=%d',
        label, checkpoint, len(state_dict), len(missing_keys), len(unexpected_keys),
    )
    if len(missing_keys) > 0:
        logging.debug('%s missing_keys: %s', label, missing_keys)
    if len(unexpected_keys) > 0:
        logging.debug('%s unexpected_keys: %s', label, unexpected_keys)
    if strict_match is True and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        raise ValueError(
            '{} {} does not match model exactly: missing_keys={} unexpected_keys={}'.format(
                label, checkpoint, len(missing_keys), len(unexpected_keys)))
    return missing_keys, unexpected_keys


def load_teacher_checkpoint(model: torch.nn.Module, checkpoint: str, strict_match: bool = True) -> Tuple[list, list]:
    return load_checkpoint_to_model(
        model,
        checkpoint,
        strict_match=strict_match,
        label='distill teacher checkpoint',
    )


def freeze_module(model: Optional[torch.nn.Module]) -> None:
    if model is None:
        return
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)


def _masked_mean(loss_per_token: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid_mask = valid_mask.to(loss_per_token.dtype)
    denom = valid_mask.sum().clamp_min(1.0)
    return (loss_per_token * valid_mask).sum() / denom


def sparse_topk_kd_loss(student_logits: torch.Tensor,
                       teacher_logits: torch.Tensor,
                       valid_mask: torch.Tensor,
                       top_k: int = 16,
                       loss_type: str = 'reverse_kl_topk',
                       temperature: float = 1.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Sparse top-k KD used by OPD-style distillation.

    reverse_kl_topk mirrors OPD only_stu support selection: the student selects
    the top-k token support, and teacher probabilities are gathered on that
    same support. forward_kl_topk selects the support from the teacher.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError('student_logits shape {} != teacher_logits shape {}'.format(
            tuple(student_logits.shape), tuple(teacher_logits.shape)))
    if loss_type not in ['reverse_kl_topk', 'forward_kl_topk']:
        raise ValueError('unsupported kd_loss {}'.format(loss_type))
    if temperature <= 0:
        raise ValueError('kd_temperature must be > 0, got {}'.format(temperature))

    vocab_size = student_logits.size(-1)
    top_k = min(max(int(top_k), 1), vocab_size)
    valid_mask = valid_mask.bool()
    if valid_mask.sum() == 0:
        zero = student_logits.sum() * 0.0
        metrics = {
            'kd_top1_agree': zero.detach(),
            'kd_topk_overlap': zero.detach(),
            'kd_token_count': zero.detach(),
        }
        return zero, metrics

    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    with torch.no_grad():
        teacher_log_probs = F.log_softmax(teacher_logits.float() / temperature, dim=-1)

    if loss_type == 'reverse_kl_topk':
        _, top_ids = torch.topk(student_log_probs.detach(), k=top_k, dim=-1)
        student_selected = torch.gather(student_log_probs, dim=-1, index=top_ids)
        teacher_selected = torch.gather(teacher_log_probs, dim=-1, index=top_ids).detach()
        student_log_norm = student_selected - torch.logsumexp(student_selected, dim=-1, keepdim=True)
        teacher_log_norm = teacher_selected - torch.logsumexp(teacher_selected, dim=-1, keepdim=True)
        student_prob = student_log_norm.exp()
        loss_per_token = (student_prob * (student_log_norm - teacher_log_norm)).sum(dim=-1)
    else:
        _, top_ids = torch.topk(teacher_log_probs, k=top_k, dim=-1)
        student_selected = torch.gather(student_log_probs, dim=-1, index=top_ids)
        teacher_selected = torch.gather(teacher_log_probs, dim=-1, index=top_ids).detach()
        student_log_norm = student_selected - torch.logsumexp(student_selected, dim=-1, keepdim=True)
        teacher_log_norm = teacher_selected - torch.logsumexp(teacher_selected, dim=-1, keepdim=True)
        teacher_prob = teacher_log_norm.exp()
        loss_per_token = (teacher_prob * (teacher_log_norm - student_log_norm)).sum(dim=-1)

    loss_per_token = torch.nan_to_num(loss_per_token, nan=0.0, posinf=0.0, neginf=0.0)
    loss = _masked_mean(loss_per_token, valid_mask) * (temperature ** 2)

    with torch.no_grad():
        student_top1 = student_logits.argmax(dim=-1)
        teacher_top1 = teacher_logits.argmax(dim=-1)
        student_topk = torch.topk(student_logits, k=top_k, dim=-1).indices
        teacher_topk = torch.topk(teacher_logits, k=top_k, dim=-1).indices
        topk_overlap = (
            student_topk.unsqueeze(dim=-1) == teacher_topk.unsqueeze(dim=-2)
        ).any(dim=-1).to(student_logits.dtype).sum(dim=-1) / float(top_k)
        token_count = valid_mask.sum().to(student_logits.dtype)
        top1_agree = ((student_top1 == teacher_top1) & valid_mask).to(student_logits.dtype).sum() / token_count.clamp_min(1.0)
        topk_overlap = _masked_mean(topk_overlap, valid_mask)
    metrics = {
        'kd_top1_agree': top1_agree.detach(),
        'kd_topk_overlap': topk_overlap.detach(),
        'kd_token_count': token_count.detach(),
    }
    return loss, metrics


def sparse_reverse_kl_topk(student_logits: torch.Tensor,
                           teacher_logits: torch.Tensor,
                           valid_mask: torch.Tensor,
                           top_k: int = 16,
                           temperature: float = 1.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    return sparse_topk_kd_loss(
        student_logits,
        teacher_logits,
        valid_mask,
        top_k=top_k,
        loss_type='reverse_kl_topk',
        temperature=temperature,
    )


def sparse_forward_kl_topk(student_logits: torch.Tensor,
                           teacher_logits: torch.Tensor,
                           valid_mask: torch.Tensor,
                           top_k: int = 16,
                           temperature: float = 1.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    return sparse_topk_kd_loss(
        student_logits,
        teacher_logits,
        valid_mask,
        top_k=top_k,
        loss_type='forward_kl_topk',
        temperature=temperature,
    )


class DistillManager:
    def __init__(self,
                 external_teacher: Optional[torch.nn.Module],
                 ema_teacher: Optional[torch.nn.Module],
                 mode: str,
                 kd_top_k: int,
                 kd_loss: str,
                 kd_weight: float,
                 ema_teacher_weight: float,
                 ema_decay: float,
                 kd_temperature: float,
                 online_start_step: int,
                 online_interval: int):
        if mode not in ['off', 'forced', 'online', 'hybrid', 'opsd']:
            raise ValueError('unsupported distill_mode {}'.format(mode))
        if mode == 'opsd' and (kd_weight <= 0 or ema_teacher_weight != 0):
            raise ValueError('distill_mode opsd requires kd_weight > 0 and ema_teacher_weight == 0')
        if mode != 'off' and kd_weight <= 0 and ema_teacher_weight <= 0:
            raise ValueError('at least one teacher weight must be > 0 for distill_mode {}'.format(mode))
        if online_start_step < 0:
            raise ValueError('online_start_step must be >= 0')
        if online_interval <= 0:
            raise ValueError('online_interval must be > 0')
        self.external_teacher = external_teacher
        self.ema_teacher = ema_teacher
        self.mode = mode
        self.kd_top_k = kd_top_k
        self.kd_loss = kd_loss
        self.kd_weight = kd_weight
        self.ema_teacher_weight = ema_teacher_weight
        self.ema_decay = ema_decay
        self.kd_temperature = kd_temperature
        self.online_start_step = online_start_step
        self.online_interval = online_interval
        self._online_warned = False
        freeze_module(self.external_teacher)
        freeze_module(self.ema_teacher)

    @property
    def enabled(self) -> bool:
        return self.mode != 'off'

    def to(self, device: torch.device) -> 'DistillManager':
        if self.external_teacher is not None:
            self.external_teacher.to(device)
            self.external_teacher.eval()
        if self.ema_teacher is not None:
            self.ema_teacher.to(device)
            self.ema_teacher.eval()
        return self

    def _online_due(self, step: int) -> bool:
        if self.mode in ['online', 'opsd']:
            return True
        return self.mode == 'hybrid' and step >= self.online_start_step and \
            (step - self.online_start_step) % self.online_interval == 0

    @torch.no_grad()
    def teacher_forward_logits(self,
                               teacher_name: str,
                               batch: dict,
                               device,
                               branch_choices=None,
                               speech_token_override: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Optional[Dict[str, torch.Tensor]]:
        if teacher_name == 'external':
            teacher_model = self.external_teacher
        elif teacher_name == 'ema':
            teacher_model = self.ema_teacher
        else:
            raise ValueError('teacher_name must be external or ema, got {}'.format(teacher_name))
        if teacher_model is None:
            return None
        teacher_model.eval()
        return teacher_model.forward_logits(
            batch,
            device,
            branch_choices=branch_choices,
            speech_token_override=speech_token_override,
        )

    @staticmethod
    def _torch_device(device) -> torch.device:
        if isinstance(device, torch.device):
            return device
        if isinstance(device, int):
            return torch.device('cuda:{}'.format(device))
        return torch.device(device)

    def _slice_batch(self, batch: dict, indices: List[int]) -> dict:
        sliced_batch = {}
        index_tensor_by_device = {}
        batch_size = len(batch['text_token'])
        for key, value in batch.items():
            if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == batch_size:
                device_key = str(value.device)
                if device_key not in index_tensor_by_device:
                    index_tensor_by_device[device_key] = torch.tensor(indices, dtype=torch.long, device=value.device)
                sliced_batch[key] = value.index_select(0, index_tensor_by_device[device_key])
            elif isinstance(value, list) and len(value) == batch_size:
                sliced_batch[key] = [value[i] for i in indices]
            else:
                sliced_batch[key] = value
        return sliced_batch

    @staticmethod
    def _clone_batch_tensors(batch: dict) -> dict:
        cloned_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                # Online rollout runs through inference-only generation.  Clone
                # the selected batch before feeding it back to the train graph.
                cloned_batch[key] = value.detach().clone()
            else:
                cloned_batch[key] = value
        return cloned_batch

    @torch.no_grad()
    def _sample_online_batch(self, student_model: torch.nn.Module, batch: dict, device, step: int) -> Optional[dict]:
        if student_model.__class__.__name__ != 'Qwen2LM':
            raise NotImplementedError('online distillation currently supports Qwen2LM/CosyVoice2 only, got {}'.format(
                student_model.__class__.__name__))
        torch_device = self._torch_device(device)
        text_token = batch['text_token']
        text_token_len = batch['text_token_len']
        speech_token_len = batch['speech_token_len']
        sampled_tokens = []
        kept_indices = []
        was_training = student_model.training
        student_model.eval()
        autocast_context = torch.amp.autocast('cuda', enabled=False) if torch_device.type == 'cuda' else nullcontext()
        try:
            with autocast_context:
                for i in range(text_token.size(0)):
                    this_text_len = max(int(text_token_len[i].item()), 1)
                    if self.mode == 'opsd':
                        min_ratio, max_ratio = 2.0, 20.0
                    else:
                        this_speech_len = max(int(speech_token_len[i].item()), 1)
                        max_ratio = max(float(this_speech_len) / float(this_text_len), 1.0)
                        min_ratio = min(2.0, max_ratio)
                    this_text = text_token[i:i + 1, :this_text_len].to(torch_device)
                    this_text_len_tensor = torch.tensor([this_text_len], dtype=torch.int32, device=torch_device)
                    prompt_text = torch.zeros(1, 0, dtype=this_text.dtype, device=torch_device)
                    prompt_text_len = torch.zeros(1, dtype=torch.int32, device=torch_device)
                    prompt_speech_token = torch.zeros(1, 0, dtype=torch.long, device=torch_device)
                    prompt_speech_token_len = torch.zeros(1, dtype=torch.int32, device=torch_device)
                    embedding = torch.zeros(0, device=torch_device)
                    tokens = []
                    for token in student_model.inference(
                            this_text,
                            this_text_len_tensor,
                            prompt_text,
                            prompt_text_len,
                            prompt_speech_token,
                            prompt_speech_token_len,
                            embedding,
                            sampling=25,
                            min_token_text_ratio=min_ratio,
                            max_token_text_ratio=max_ratio,
                            uuid='distill_{}_{}'.format(step, i)):
                        if torch.is_tensor(token):
                            token = int(token.item())
                        else:
                            token = int(token)
                        if token < student_model.speech_token_size:
                            tokens.append(token)
                    if len(tokens) > 0:
                        sampled_tokens.append(torch.tensor(tokens, dtype=torch.long, device=torch_device))
                        kept_indices.append(i)
        finally:
            if torch_device.type == 'cuda' and hasattr(torch, 'clear_autocast_cache'):
                torch.clear_autocast_cache()
            if was_training is True:
                student_model.train()

        if len(sampled_tokens) == 0:
            return None
        online_batch = self._slice_batch(batch, kept_indices)
        online_batch['speech_token'] = pad_sequence(sampled_tokens, batch_first=True, padding_value=0)
        online_batch['speech_token_len'] = torch.tensor(
            [tokens.numel() for tokens in sampled_tokens], dtype=torch.int32, device=torch_device)
        return self._clone_batch_tensors(online_batch)

    def _add_teacher_kd(self,
                        loss_dict: Dict[str, torch.Tensor],
                        total_loss: torch.Tensor,
                        prefix: str,
                        student_out: Dict[str, torch.Tensor],
                        teacher_model: Optional[torch.nn.Module],
                        valid_mask: torch.Tensor,
                        batch: dict,
                        device,
                        branch_choices,
                        weight: float) -> torch.Tensor:
        if teacher_model is None or weight <= 0:
            loss_dict['{}kd_loss'.format(prefix)] = total_loss.detach() * 0.0
            loss_dict['{}speech_kd_loss'.format(prefix)] = total_loss.detach() * 0.0
            return total_loss
        with torch.no_grad():
            teacher_out = teacher_model.forward_logits(batch, device, branch_choices=branch_choices)
        kd_loss, kd_metrics = sparse_topk_kd_loss(
            student_out['logits'],
            teacher_out['logits'],
            valid_mask,
            top_k=self.kd_top_k,
            loss_type=self.kd_loss,
            temperature=self.kd_temperature,
        )
        total_loss = total_loss + weight * kd_loss
        loss_dict['{}kd_loss'.format(prefix)] = kd_loss.detach()
        loss_dict['{}kd_weighted'.format(prefix)] = (weight * kd_loss).detach()
        for key, value in kd_metrics.items():
            loss_dict['{}{}'.format(prefix, key)] = value
        speech_mask = student_out.get('speech_mask')
        if speech_mask is not None and speech_mask.any():
            speech_kd_loss, speech_kd_metrics = sparse_topk_kd_loss(
                student_out['logits'],
                teacher_out['logits'],
                speech_mask,
                top_k=self.kd_top_k,
                loss_type=self.kd_loss,
                temperature=self.kd_temperature,
            )
            loss_dict['{}speech_kd_loss'.format(prefix)] = speech_kd_loss.detach()
            for key, value in speech_kd_metrics.items():
                loss_dict['{}speech_{}'.format(prefix, key)] = value
        else:
            loss_dict['{}speech_kd_loss'.format(prefix)] = total_loss.detach() * 0.0
        return total_loss

    def _add_opsd_teacher_kd(self,
                             loss_dict: Dict[str, torch.Tensor],
                             total_loss: torch.Tensor,
                             student_out: Dict[str, torch.Tensor],
                             teacher_model: Optional[torch.nn.Module],
                             batch: dict,
                             device,
                             weight: float) -> torch.Tensor:
        zero = total_loss.detach() * 0.0
        if teacher_model is None or weight <= 0:
            loss_dict['opsd_kd_loss'] = zero
            loss_dict['opsd_speech_kd_loss'] = zero
            return total_loss
        with torch.no_grad():
            teacher_out = teacher_model.forward_logits(
                batch,
                device,
                branch_choices=[False] * batch['speech_token'].size(0),
                speech_token_override=(batch['speech_token'], batch['speech_token_len']),
                prompt_conditioned=True,
            )

        student_mask = student_out.get('speech_mask', student_out['valid_mask']).bool()
        teacher_mask = teacher_out.get('speech_mask', teacher_out['valid_mask']).bool()
        student_count = int(student_mask.sum().item())
        teacher_count = int(teacher_mask.sum().item())
        if student_count != teacher_count:
            raise ValueError('OPSD student/teacher speech token count mismatch: student={} teacher={}'.format(
                student_count, teacher_count))
        if student_count == 0:
            loss_dict['opsd_kd_loss'] = zero
            loss_dict['opsd_speech_kd_loss'] = zero
            loss_dict['opsd_kd_token_count'] = zero
            return total_loss

        student_logits = student_out['logits'][student_mask].unsqueeze(0)
        teacher_logits = teacher_out['logits'][teacher_mask].unsqueeze(0)
        active_mask = torch.ones(
            student_logits.shape[:2], dtype=torch.bool, device=student_logits.device)
        kd_loss, kd_metrics = sparse_topk_kd_loss(
            student_logits,
            teacher_logits,
            active_mask,
            top_k=self.kd_top_k,
            loss_type=self.kd_loss,
            temperature=self.kd_temperature,
        )
        total_loss = total_loss + weight * kd_loss
        loss_dict['opsd_kd_loss'] = kd_loss.detach()
        loss_dict['opsd_speech_kd_loss'] = kd_loss.detach()
        loss_dict['opsd_kd_weighted'] = (weight * kd_loss).detach()
        for key, value in kd_metrics.items():
            loss_dict['opsd_{}'.format(key)] = value
            loss_dict['opsd_speech_{}'.format(key)] = value
        return total_loss

    def forward(self, model: torch.nn.Module, batch: dict, device: torch.device, info_dict: dict) -> Dict[str, torch.Tensor]:
        student_model = unwrap_model(model)
        if not hasattr(student_model, 'forward_logits'):
            raise ValueError('distillation requires model.forward_logits, got {}'.format(student_model.__class__.__name__))

        if self.mode == 'opsd':
            zero = next(student_model.parameters()).sum() * 0.0
            loss_dict = {
                'loss': zero,
                'acc': zero.detach(),
                'ce_loss': zero.detach(),
                'speech_ce_loss': zero.detach(),
                'external_kd_loss': zero.detach(),
                'ema_kd_loss': zero.detach(),
            }
            total_loss = zero
            online_batch = self._sample_online_batch(student_model, batch, device, int(info_dict.get('step', 0)))
            if online_batch is not None:
                online_student_out = student_model.forward_logits(
                    online_batch,
                    device,
                    branch_choices=[False] * online_batch['speech_token'].size(0),
                    speech_token_override=(online_batch['speech_token'], online_batch['speech_token_len']),
                )
                total_loss = self._add_opsd_teacher_kd(
                    loss_dict, total_loss, online_student_out, self.external_teacher,
                    online_batch, device, self.kd_weight)
                loss_dict['ce_loss'] = online_student_out.get('loss_ce', online_student_out['loss']).detach()
                loss_dict['speech_ce_loss'] = online_student_out.get(
                    'speech_ce_loss', online_student_out['loss'].detach() * 0.0)
                loss_dict['acc'] = online_student_out['acc']
                loss_dict['online_sample_token_count'] = online_batch['speech_token_len'].sum().detach()
                loss_dict['online_sample_batch_size'] = torch.tensor(
                    online_batch['speech_token'].size(0), dtype=total_loss.dtype, device=total_loss.device)
            else:
                loss_dict['online_sample_token_count'] = zero.detach()
                loss_dict['online_sample_batch_size'] = zero.detach()
                loss_dict['opsd_kd_loss'] = zero.detach()
                loss_dict['opsd_speech_kd_loss'] = zero.detach()
            loss_dict['loss'] = total_loss
            loss_dict['kd_loss'] = total_loss.detach()
            return loss_dict

        student_out = student_model.forward_logits(batch, device)
        base_train_loss = student_out['loss'] if self.mode != 'online' else student_out['loss'] * 0.0
        loss_dict = {
            'loss': base_train_loss,
            'acc': student_out['acc'],
            'ce_loss': student_out.get('loss_ce', student_out['loss']).detach(),
            'speech_ce_loss': student_out.get('speech_ce_loss', student_out['loss'].detach() * 0.0),
        }
        total_loss = base_train_loss

        branch_choices = student_out.get('branch_choices')
        valid_mask = student_out['valid_mask']
        if self.mode in ['forced', 'hybrid']:
            total_loss = self._add_teacher_kd(
                loss_dict, total_loss, 'external_', student_out, self.external_teacher, valid_mask,
                batch, device, branch_choices, self.kd_weight)
            total_loss = self._add_teacher_kd(
                loss_dict, total_loss, 'ema_', student_out, self.ema_teacher, valid_mask,
                batch, device, branch_choices, self.ema_teacher_weight)
        else:
            loss_dict['external_kd_loss'] = total_loss.detach() * 0.0
            loss_dict['ema_kd_loss'] = total_loss.detach() * 0.0

        if self._online_due(int(info_dict.get('step', 0))):
            online_batch = self._sample_online_batch(student_model, batch, device, int(info_dict.get('step', 0)))
            if online_batch is not None:
                online_student_out = student_model.forward_logits(
                    online_batch,
                    device,
                    speech_token_override=(online_batch['speech_token'], online_batch['speech_token_len']),
                )
                online_branch_choices = online_student_out.get('branch_choices')
                online_kd_mask = online_student_out.get('speech_mask')
                if online_kd_mask is None:
                    online_kd_mask = online_student_out['valid_mask']
                total_loss = self._add_teacher_kd(
                    loss_dict, total_loss, 'online_external_', online_student_out, self.external_teacher,
                    online_kd_mask, online_batch, device, online_branch_choices, self.kd_weight)
                total_loss = self._add_teacher_kd(
                    loss_dict, total_loss, 'online_ema_', online_student_out, self.ema_teacher,
                    online_kd_mask, online_batch, device, online_branch_choices, self.ema_teacher_weight)
                loss_dict['online_sample_token_count'] = online_batch['speech_token_len'].sum().detach()
                loss_dict['online_sample_batch_size'] = torch.tensor(
                    online_batch['speech_token'].size(0), dtype=total_loss.dtype, device=total_loss.device)
            else:
                zero = total_loss.detach() * 0.0
                loss_dict['online_sample_token_count'] = zero
                loss_dict['online_sample_batch_size'] = zero
                loss_dict['online_external_kd_loss'] = zero
                loss_dict['online_external_speech_kd_loss'] = zero
                loss_dict['online_ema_kd_loss'] = zero
                loss_dict['online_ema_speech_kd_loss'] = zero

        loss_dict['loss'] = total_loss
        loss_dict['kd_loss'] = (total_loss - base_train_loss).detach()
        return loss_dict

    @torch.no_grad()
    def update_ema(self, model: torch.nn.Module) -> None:
        if self.ema_teacher is None or self.ema_teacher_weight <= 0:
            return
        student_model = unwrap_model(model)
        self.ema_teacher.eval()
        for ema_param, student_param in zip(self.ema_teacher.parameters(), student_model.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(student_param.detach().data, alpha=1.0 - self.ema_decay)
        for ema_buffer, student_buffer in zip(self.ema_teacher.buffers(), student_model.buffers()):
            ema_buffer.copy_(student_buffer.detach())


class TeacherManager(DistillManager):
    """Named manager requested by the OPD plan; keeps DistillManager behavior."""
    pass
