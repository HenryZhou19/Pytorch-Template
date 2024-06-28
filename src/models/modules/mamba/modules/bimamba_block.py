# Mamba Copyright (c) 2023, Tri Dao, Albert Gu.
# BiMamba from https://github.com/hustvl/Vim (Vision Mamba)
# Modified by Jiaheng Zhou: https://github.com/HenryZhou19/Pytorch-Template

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm.utils.generation import InferenceParams

from .bimamba_inner_interface import *
from .mamba_block import MambaBlock
from .mamba_inner_interface import *
from .utils import *


class BiMambaBlock(MambaBlock):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_index=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        if_divide_out=False,
        # init_layer_scale=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            conv_bias=conv_bias,
            bias=bias,
            use_fast_path=use_fast_path,
            layer_index=layer_index,
            device=device,
            dtype=dtype,
            )
        
        self.bimamba_type = bimamba_type
        self.if_divide_out = if_divide_out
        
        # self.init_layer_scale = init_layer_scale
        # if init_layer_scale is not None:
        #     self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)
        
        # Reverse part
        if bimamba_type == "v1":
            self.A_log_rev = init_A_log(self.d_state, self.d_inner, device)
        elif bimamba_type == "v2":
            self.conv1d_rev = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            
            self.x_proj_rev = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_rev = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            
            self.A_log_rev = init_A_log(self.d_state, self.d_inner, device)
            
            self.D_rev = init_D(self.d_inner, device)
            
    def _forward_slow_path(self, xz, A, conv_state, ssm_state, seqlen):
        assert self.bimamba_type == "none", "BiMamba does not support slow path currently"
        return super()._forward_slow_path(xz, A, conv_state, ssm_state, seqlen)
            
    def _forward_fast_path(self, xz, A):
        if self.bimamba_type == "v1":
            A_rev = -torch.exp(self.A_log_rev.float())
            out = bimamba_v1_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                A,
                A_rev,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            out = rearrange(out, "b d l -> b l d")
        elif self.bimamba_type == "v2":
            A_rev = -torch.exp(self.A_log_rev.float())
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            out_b = mamba_inner_fn(
                xz.flip([-1]),
                self.conv1d_rev.weight,
                self.conv1d_rev.bias,
                self.x_proj_rev.weight,
                self.dt_proj_rev.weight,
                A_rev,
                None,
                None,
                self.D_rev.float(),
                delta_bias=self.dt_proj_rev.bias.float(),
                delta_softplus=True,
            )
            # F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
            if not self.if_divide_out:
                out = rearrange(out + out_b.flip([-1]), "b d l -> b l d")
            else:
                out = rearrange(out + out_b.flip([-1]), "b d l -> b l d") / 2
        else:
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            out = rearrange(out, "b d l -> b l d")
            
        out = F.linear(out, self.out_proj.weight, self.out_proj.bias)
        return out

    def forward(self, hidden_states, inference_params: InferenceParams=None):
        """
        hidden_states: (B, L, d)
        inference_params: "InferenceParams" are passed to the main model in order to efficienly calculate and store the context during inference.
        Returns:
            out: (B, L, d)
        """
        out = super().forward(hidden_states, inference_params)
        # if self.init_layer_scale is not None:
        #         out = out * self.gamma    
        return out

    #def step(self, hidden_states, conv_state, ssm_state):
    # XXX: not used in BiMamba? so not overriden

    # def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
    # XXX: not used in BiMamba? so not overriden

