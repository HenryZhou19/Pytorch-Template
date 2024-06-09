# Mamba Copyright (c) 2023, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn
from einops import repeat

__all__ = [
    'init_dt_proj',
    'init_A_log',
    'init_D',
    'init_mamba_weights',
    ]

def init_dt_proj(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, factory_kwargs):
    dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

    # Initialize special dt projection to preserve variance at initialization
    dt_init_std = dt_rank**-0.5 * dt_scale
    if dt_init == 'constant':
        nn.init.constant_(dt_proj.weight, dt_init_std)
    elif dt_init == 'random':
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
        'n -> d n',
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


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def init_mamba_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
    verbose=False,
    ):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, '_no_reinit', False):
                nn.init.zeros_(module.bias)
            else:
                if verbose:
                    print(f'Skipping reinitialization of {module}\'s bias')
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ['out_proj.weight', 'fc2.weight']:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)
                    