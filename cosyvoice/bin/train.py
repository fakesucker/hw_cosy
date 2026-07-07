# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import argparse
import datetime
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from copy import deepcopy
import os
import torch
import torch.distributed as dist
import deepspeed

from hyperpyyaml import load_hyperpyyaml

from torch.distributed.elastic.multiprocessing.errors import record

from cosyvoice.utils.losses import DPOLoss
from cosyvoice.utils.distill_utils import DistillManager, extract_model_state_dict, load_teacher_checkpoint
from cosyvoice.utils.executor import Executor
from cosyvoice.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config)


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser.add_argument('--model', required=True, help='model which will be trained')
    parser.add_argument('--ref_model', required=False, help='ref model used in dpo')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--qwen_pretrain_path', required=False, help='qwen pretrain path')
    parser.add_argument('--onnx_path', required=False, help='onnx path, which is required for online feature extraction')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--dpo',
                        action='store_true',
                        default=False,
                        help='Use Direct Preference Optimization')
    parser.add_argument('--distill_mode',
                        default='off',
                        choices=['off', 'forced', 'online', 'hybrid', 'opsd'],
                        help='OPD-style distillation mode. online samples student rollouts on interval; hybrid combines forced and online KD.')
    parser.add_argument('--teacher_checkpoint',
                        default=None,
                        help='External teacher checkpoint used by OPD-style distillation')
    parser.add_argument('--kd_top_k',
                        default=16,
                        type=int,
                        help='Top-k support size for sparse KD')
    parser.add_argument('--kd_loss',
                        default='reverse_kl_topk',
                        choices=['reverse_kl_topk', 'forward_kl_topk'],
                        help='Sparse KD loss type')
    parser.add_argument('--kd_weight',
                        default=0.2,
                        type=float,
                        help='External teacher KD loss weight')
    parser.add_argument('--ema_teacher_weight',
                        default=0.05,
                        type=float,
                        help='EMA teacher KD loss weight')
    parser.add_argument('--ema_decay',
                        default=0.999,
                        type=float,
                        help='EMA teacher decay')
    parser.add_argument('--kd_temperature',
                        default=1.0,
                        type=float,
                        help='KD softmax temperature')
    parser.add_argument('--online_start_step',
                        default=2000,
                        type=int,
                        help='Step at which hybrid online distillation should start')
    parser.add_argument('--online_interval',
                        default=4,
                        type=int,
                        help='Hybrid online distillation interval')
    parser.add_argument('--debug_max_steps',
                        default=0,
                        type=int,
                        help='Stop after this many optimizer steps when > 0')
    parser.add_argument('--max_train_steps',
                        default=0,
                        type=int,
                        help='Stop after this many optimizer steps when > 0')
    parser.add_argument('--max_epoch',
                        default=None,
                        type=int,
                        help='Override train_conf.max_epoch when provided')
    parser.add_argument('--save_per_step',
                        default=None,
                        type=int,
                        help='Override train_conf.save_per_step; >0 enables step checkpoints, <=0 disables them')
    parser.add_argument('--log_interval',
                        default=None,
                        type=int,
                        help='Override train_conf.log_interval; 1 prints shell logs every batch')
    parser.add_argument('--skip_cv_on_step_save',
                        action='store_true',
                        default=False,
                        help='Save step checkpoints directly without running CV first')
    parser.add_argument('--max_frames_in_batch',
                        default=None,
                        type=int,
                        help='Override data_pipeline batch.max_frames_in_batch when set')
    parser.add_argument('--accum_grad',
                        default=None,
                        type=int,
                        help='Override train_conf.accum_grad when provided')
    parser.add_argument(
        '--train_branch_mode',
        default=None,
        choices=['auto', 'unistream', 'bistream'],
        help='LLM train sequence layout (yaml train_branch_mode): auto=random 50%% bistream when ratio ok; '
             'unistream=full sequence; bistream=always use bi-stream blocks when speech/text ratio ok',
    )
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')
    parser.add_argument('--timeout',
                        default=60,
                        type=int,
                        help='timeout (in seconds) of cosyvoice_join.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def torch_load_checkpoint(checkpoint):
    try:
        return torch.load(checkpoint, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(checkpoint, map_location='cpu')


def override_batch_frames(configs, max_frames_in_batch):
    if max_frames_in_batch is None:
        return
    if max_frames_in_batch <= 0:
        raise ValueError('--max_frames_in_batch must be > 0 when provided')
    for pipeline_key in ['data_pipeline', 'data_pipeline_gan']:
        for processor in configs.get(pipeline_key, []):
            if getattr(getattr(processor, 'func', None), '__name__', None) == 'batch':
                processor.keywords['max_frames_in_batch'] = max_frames_in_batch


@record
def main():
    args = get_args()
    # import pdb; pdb.set_trace()
    if args.onnx_path is not None:
        os.environ['onnx_path'] = args.onnx_path
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    # gan train has some special initialization logic
    gan = True if args.model == 'hifigan' else False
    if args.distill_mode == 'opsd':
        if args.train_branch_mode is None:
            args.train_branch_mode = 'unistream'
        elif args.train_branch_mode != 'unistream':
            raise ValueError('--distill_mode opsd requires --train_branch_mode unistream')
    if args.distill_mode != 'off':
        if args.model != 'llm':
            raise ValueError('--distill_mode only supports --model llm, got {}'.format(args.model))
        if args.dpo is True:
            raise ValueError('--distill_mode and --dpo cannot be enabled together in this trainer')
        if args.train_engine != 'torch_ddp':
            raise ValueError('--distill_mode currently supports torch_ddp only, got {}'.format(args.train_engine))
        if args.kd_weight < 0 or args.ema_teacher_weight < 0:
            raise ValueError('kd_weight and ema_teacher_weight must be non-negative')
        if args.kd_weight == 0 and args.ema_teacher_weight == 0:
            raise ValueError('at least one of kd_weight or ema_teacher_weight must be > 0 when distill_mode != off')
        if args.distill_mode == 'opsd' and (args.kd_weight <= 0 or args.ema_teacher_weight != 0):
            raise ValueError('--distill_mode opsd requires --kd_weight > 0 and --ema_teacher_weight 0')
        if args.teacher_checkpoint is None or not os.path.exists(args.teacher_checkpoint):
            raise ValueError('--teacher_checkpoint is required and must exist when distill_mode != off')
        if args.kd_top_k <= 0:
            raise ValueError('--kd_top_k must be > 0')
        if args.kd_temperature <= 0:
            raise ValueError('--kd_temperature must be > 0')
        if args.ema_decay < 0 or args.ema_decay >= 1:
            raise ValueError('--ema_decay must be in [0, 1)')
        if args.online_start_step < 0:
            raise ValueError('--online_start_step must be >= 0')
        if args.online_interval <= 0:
            raise ValueError('--online_interval must be > 0')
    if args.save_per_step is not None and args.save_per_step < -1:
        raise ValueError('--save_per_step must be >= -1 when provided')
    if args.log_interval is not None and args.log_interval <= 0:
        raise ValueError('--log_interval must be > 0 when provided')
    if args.max_train_steps < 0:
        raise ValueError('--max_train_steps must be >= 0')
    if args.max_epoch is not None and args.max_epoch <= 0:
        raise ValueError('--max_epoch must be > 0 when provided')
    if args.accum_grad is not None and args.accum_grad <= 0:
        raise ValueError('--accum_grad must be > 0 when provided')

    override_dict = {k: None for k in ['llm', 'flow', 'hift', 'hifigan'] if k != args.model}
    if gan is True:
        override_dict.pop('hift')
    if args.qwen_pretrain_path is not None:
        override_dict['qwen_pretrain_path'] = args.qwen_pretrain_path
    if args.train_branch_mode is not None:
        override_dict['train_branch_mode'] = args.train_branch_mode
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)
    override_batch_frames(configs, args.max_frames_in_batch)
    if gan is True:
        configs['train_conf'] = configs['train_conf_gan']
    arg_dict = vars(args).copy()
    save_per_step_override = arg_dict.pop('save_per_step')
    log_interval_override = arg_dict.pop('log_interval')
    max_epoch_override = arg_dict.pop('max_epoch')
    accum_grad_override = arg_dict.pop('accum_grad')
    arg_dict.pop('max_frames_in_batch')
    configs['train_conf'].update(arg_dict)
    if args.max_train_steps > 0:
        configs['train_conf']['debug_max_steps'] = args.max_train_steps
    if max_epoch_override is not None:
        configs['train_conf']['max_epoch'] = max_epoch_override
    if accum_grad_override is not None:
        configs['train_conf']['accum_grad'] = accum_grad_override
    if save_per_step_override is not None:
        configs['train_conf']['save_per_step'] = save_per_step_override
    if log_interval_override is not None:
        configs['train_conf']['log_interval'] = log_interval_override

    # Init env for ddp
    _, local_rank, _ = init_distributed(args)

    # Get dataset & dataloader
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, gan, args.dpo)

    # Do some sanity checks and save config to arsg.model_dir
    configs = check_modify_and_save_config(args, configs)
    if args.model == 'llm' and hasattr(configs['llm'], 'branch_mode'):
        logging.info('LLM train_branch_mode=%s mix_ratio=%s', configs['llm'].branch_mode, configs['llm'].mix_ratio)

    # Tensorboard summary
    writer = init_summarywriter(args)

    # load checkpoint
    if args.dpo is True:
        configs[args.model].forward = configs[args.model].forward_dpo
    model = configs[args.model]
    start_step, start_epoch = 0, -1
    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            checkpoint_dict = torch_load_checkpoint(args.checkpoint)
            state_dict = extract_model_state_dict(checkpoint_dict, args.checkpoint)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            logging.info(
                'Loaded student checkpoint %s, tensor_keys=%d missing_keys=%d unexpected_keys=%d',
                args.checkpoint, len(state_dict), len(missing_keys), len(unexpected_keys),
            )
            if args.distill_mode != 'off' and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
                raise ValueError(
                    'student checkpoint {} does not match model exactly under distill_mode={}: '
                    'missing_keys={} unexpected_keys={}'.format(
                        args.checkpoint, args.distill_mode, len(missing_keys), len(unexpected_keys)))
            if isinstance(checkpoint_dict, dict):
                if 'step' in checkpoint_dict:
                    start_step = checkpoint_dict['step']
                if 'epoch' in checkpoint_dict:
                    start_epoch = checkpoint_dict['epoch']
        else:
            logging.warning('checkpoint {} do not exsist!'.format(args.checkpoint))

    # OPD-style distillation teachers. They are frozen single-GPU modules, not DDP-wrapped.
    distill_manager = None
    if args.distill_mode != 'off':
        # Avoid deepcopy issues if online speech token extractor was already created.
        model_for_copy = configs[args.model]
        detached_attrs = {}
        for attr in ['_speech_token_extractor', 'speech_token_extractor']:
            if hasattr(model_for_copy, attr):
                detached_attrs[attr] = getattr(model_for_copy, attr)
                setattr(model_for_copy, attr, None)
        external_teacher = deepcopy(model_for_copy) if args.kd_weight > 0 else None
        ema_teacher = deepcopy(model_for_copy) if args.ema_teacher_weight > 0 else None
        for attr, value in detached_attrs.items():
            setattr(model_for_copy, attr, value)
        if external_teacher is not None:
            load_teacher_checkpoint(external_teacher, args.teacher_checkpoint)
        distill_manager = DistillManager(
            external_teacher=external_teacher,
            ema_teacher=ema_teacher,
            mode=args.distill_mode,
            kd_top_k=args.kd_top_k,
            kd_loss=args.kd_loss,
            kd_weight=args.kd_weight,
            ema_teacher_weight=args.ema_teacher_weight,
            ema_decay=args.ema_decay,
            kd_temperature=args.kd_temperature,
            online_start_step=args.online_start_step,
            online_interval=args.online_interval,
        ).to(torch.device('cuda:{}'.format(local_rank)))
        logging.info(
            'Distillation enabled mode=%s kd_loss=%s kd_top_k=%s kd_weight=%s ema_teacher_weight=%s ema_decay=%s',
            args.distill_mode, args.kd_loss, args.kd_top_k, args.kd_weight, args.ema_teacher_weight, args.ema_decay,
        )

    # Dispatch model from cpu to gpu
    model = wrap_cuda_model(args, model)

    # Get optimizer & scheduler
    model, optimizer, scheduler, optimizer_d, scheduler_d = init_optimizer_and_scheduler(args, configs, model, gan)
    scheduler.set_step(start_step)
    if scheduler_d is not None:
        scheduler_d.set_step(start_step)

    # Save init checkpoints
    info_dict = deepcopy(configs['train_conf'])
    info_dict['step'] = start_step
    info_dict['epoch'] = start_epoch
    save_model(model, 'init', info_dict)

    # DPO related
    if args.dpo is True:
        # NOTE: onnxruntime InferenceSession is not pickle/deepcopy-able.
        # When --onnx_path is provided, model may carry speech_token_extractor.
        # Temporarily detach it before deepcopy to build ref_model safely.
        _speech_token_extractor = None
        if hasattr(configs[args.model], 'speech_token_extractor'):
            _speech_token_extractor = configs[args.model].speech_token_extractor
            configs[args.model].speech_token_extractor = None
        ref_model = deepcopy(configs[args.model])
        if _speech_token_extractor is not None:
            configs[args.model].speech_token_extractor = _speech_token_extractor
        state_dict = torch_load_checkpoint(args.ref_model)
        ref_model.load_state_dict(state_dict, strict=False)
        dpo_loss = DPOLoss(beta=0.01, label_smoothing=0.0, ipo=False)
        # NOTE maybe it is not needed to wrap ref_model as ddp because its parameter is not updated
        ref_model = wrap_cuda_model(args, ref_model)
        ref_model.eval()
    else:
        ref_model, dpo_loss = None, None

    # Get executor
    executor = Executor(gan=gan, ref_model=ref_model, dpo_loss=dpo_loss, distill_manager=distill_manager)
    executor.step = start_step

    # Init scaler, used for pytorch amp mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    print('start step {} start epoch {}'.format(start_step, start_epoch))

    # Start training loop
    for epoch in range(start_epoch + 1, info_dict['max_epoch']):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        dist.barrier()
        group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        if gan is True:
            executor.train_one_epoc_gan(model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader,
                                        writer, info_dict, scaler, group_join)
        else:
            executor.train_one_epoc(model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join, ref_model=ref_model)
        dist.destroy_process_group(group_join)
        if info_dict.get('debug_stop', False) is True:
            break


if __name__ == '__main__':
    main()
