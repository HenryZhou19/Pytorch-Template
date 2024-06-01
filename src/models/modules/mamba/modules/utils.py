# Mamba Copyright (c) 2023, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn
from einops import repeat


def init_dt_proj(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, factory_kwargs):
    dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

    # Initialize special dt projection to preserve variance at initialization
    dt_init_std = dt_rank**-0.5 * dt_scale
    if dt_init == "constant":
        nn.init.constant_(dt_proj.weight, dt_init_std)
    elif dt_init == "random":
        nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
    else:
        raise NotImplementedError

    # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
    dt = torch.exp(
        torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    ).clamp(min=dt_init_floor)
    # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    with torch.no_grad():
        dt_proj.bias.copy_(inv_dt)
    # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
    dt_proj.bias._no_reinit = True
    
    return dt_proj


def init_A_log(d_state, d_inner, device):
    A = repeat(
        torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
        "n -> d n",
        d=d_inner,
        ).contiguous()
    A_log = torch.log(A)  # Keep A_log in fp32
    A_log = nn.Parameter(A_log)
    A_log._no_weight_decay = True
    return A_log


def init_D(d_inner, device):
    D = nn.Parameter(torch.ones(d_inner, device=device))  # Keep in fp32
    D._no_weight_decay = True
    return D