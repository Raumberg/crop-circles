from __future__ import absolute_import, division, print_function

import math
import os
import typing as ty
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from optimizers import AdaBelief, RAdam


class Lambda(nn.Module):
    def __init__(self, f: ty.Callable) -> None:
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)

def setup_optimizer(
    optimizer: str,
    parameter_groups,
    lr: float,
    weight_decay: float,
) -> optim.Optimizer:
    Optimizer = {
        'adabelief': AdaBelief,
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'radam': RAdam,
        'sgd': optim.SGD,
    }[optimizer]
    momentum = (0.9,) if Optimizer is optim.SGD else ()
    return Optimizer(parameter_groups, lr, *momentum, weight_decay=weight_decay)

def setup_lr_schedule(
    optimizer: optim.Optimizer,
    lr: float,
    epoch_size: int,
    lr_schedule: ty.Optional[ty.Dict[str, ty.Any]],
) -> ty.Tuple[
    ty.Optional[optim.lr_scheduler._LRScheduler],
    ty.Dict[str, ty.Any],
    ty.Optional[int],
]:
    if lr_schedule is None:
        lr_schedule = {'type': 'constant'}
    lr_scheduler = None
    n_warmup_steps = None
    if lr_schedule['type'] in ['transformer', 'linear_warmup']:
        n_warmup_steps = (
            lr_schedule['n_warmup_steps']
            if 'n_warmup_steps' in lr_schedule
            else lr_schedule['n_warmup_epochs'] * epoch_size
        )
    elif lr_schedule['type'] == 'cyclic':
        lr_scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=lr,
            max_lr=lr_schedule['max_lr'],
            step_size_up=lr_schedule['n_epochs_up'] * epoch_size,
            step_size_down=lr_schedule['n_epochs_down'] * epoch_size,
            mode=lr_schedule['mode'],
            gamma=lr_schedule.get('gamma', 1.0),
            cycle_momentum=False,
        )
    return lr_scheduler, lr_schedule, n_warmup_steps


def load_swa_state_dict(model: nn.Module, swa_model: optim.swa_utils.AveragedModel):
    state_dict = deepcopy(swa_model.state_dict())
    del state_dict['n_averaged']
    model.load_state_dict({k[len('module.') :]: v for k, v in state_dict.items()})


def get_epoch_parameters(
    train_size: int, batch_size: ty.Union[int, str]
) -> ty.Tuple[int, int]:
    if isinstance(batch_size, str):
        if batch_size == 'v3':
            batch_size = (
                256 if train_size < 50000 else 512 if train_size < 100000 else 1024
            )
        elif batch_size == 'v1':
            batch_size = (
                16
                if train_size < 1000
                else 32
                if train_size < 10000
                else 64
                if train_size < 50000
                else 128
                if train_size < 100000
                else 256
                if train_size < 200000
                else 512
                if train_size < 500000
                else 1024
            )
        elif batch_size == 'v2':
            batch_size = (
                512 if train_size < 100000 else 1024 if train_size < 500000 else 2048
            )
    return batch_size, math.ceil(train_size / batch_size)  # type: ignore[code]


def get_linear_warmup_lr(lr: float, n_warmup_steps: int, step: int) -> float:
    assert step > 0, "1-based enumeration of steps is expected"
    return min(lr, step / (n_warmup_steps + 1) * lr)


def get_manual_lr(schedule: ty.List[float], epoch: int) -> float:
    assert epoch > 0, "1-based enumeration of epochs is expected"
    return schedule[min(epoch, len(schedule)) - 1]


def get_transformer_lr(scale: float, d: int, n_warmup_steps: int, step: int) -> float:
    return scale * d ** -0.5 * min(step ** -0.5, step * n_warmup_steps ** -1.5)

def learn(model, optimizer, loss_fn, step, batch, star) -> ty.Tuple[Tensor, ty.Any]:
    model.train()
    optimizer.zero_grad()
    out = step(batch)
    loss = loss_fn(*out) if star else loss_fn(out)
    loss.backward()
    optimizer.step()
    return loss, out

def tensor(x) -> torch.Tensor:
    assert isinstance(x, torch.Tensor)
    return ty.cast(torch.Tensor, x)

def get_n_parameters(m: nn.Module):
    return sum(x.numel() for x in m.parameters() if x.requires_grad)

def get_mlp_n_parameters(units: ty.List[int]):
    x = 0
    for a, b in zip(units, units[1:]):
        x += a * b + b
    return x

def get_lr(optimizer: optim.Optimizer) -> float:
    return next(iter(optimizer.param_groups))['lr']

def set_lr(optimizer: optim.Optimizer, lr: float) -> None:
    for x in optimizer.param_groups:
        x['lr'] = lr

def get_device() -> torch.device:
    return torch.device('cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')


@torch.no_grad()
def get_gradient_norm_ratios(m: nn.Module):
    return {
        k: v.grad.norm() / v.norm()
        for k, v in m.named_parameters()
        if v.grad is not None
    }

def is_oom_exception(err: RuntimeError) -> bool:
    return any(
        x in str(err)
        for x in [
            'CUDA out of memory',
            'CUBLAS_STATUS_ALLOC_FAILED',
            'CUDA error: out of memory',
        ]
    )

