# Mamba Copyright (c) 2023, Tri Dao, Albert Gu.
# BiMamba from https://github.com/hustvl/Vim (Vision Mamba)
# Modified by Jiaheng Zhou: https://github.com/HenryZhou19/Pytorch-Template

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.amp import custom_bwd, custom_fwd

try:
    import causal_conv1d_cuda
except ImportError:
    causal_conv1d_cuda = None

import selective_scan_cuda

from .mamba_inner_interface import _prepare_delta_B_C_D

__all__ = [
    'bimamba_v1_inner_fn',  # for bimamba v1
    ]

class BiMambaV1InnerFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                A, A_rev, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
        """
        xz: (batch, 2*dim, seqlen) i.e. (B, 2*D, L)
        """
        assert A.is_complex() == A_rev.is_complex(), "A and A_rev should have the same dtype"
        
        assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        x, z = xz.chunk(2, dim=1)
        
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
            x, conv1d_weight, conv1d_bias, None, None, None, True
            )
        
        # We're being very careful here about the layout, to avoid extra transposes.
        # We want delta to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        
        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        
        delta, B, C, D = _prepare_delta_B_C_D(x_dbl, delta_proj_weight, delta_rank, d_state, A, B, C, D, B_proj_bias, C_proj_bias, L)
        
        out, scan_intermediates, y = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        
        # Reverse part
        out_rev, scan_intermediates_rev, y_rev = selective_scan_cuda.fwd(
            conv1d_out.flip([-1]), delta.flip([-1]), A_rev, B.flip([-1]), C.flip([-1]), D, z.flip([-1]), delta_bias, delta_softplus,
        )

        y = y + y_rev.flip([-1])

        ctx.delta_softplus = delta_softplus
        ctx.checkpoint_lvl = checkpoint_lvl
        if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
            conv1d_out, delta = None, None
        ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
                              delta_proj_weight, conv1d_out, delta,
                              A, A_rev, B, C, D, delta_bias, scan_intermediates, scan_intermediates_rev, out, out_rev)
        return y

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, dy):
        """
        dy: (batch, dim, seqlen) i.e. (B, D, L)
        """
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, 
         conv1d_out, delta, A, A_rev, B, C, D, delta_bias, scan_intermediates, scan_intermediates_rev, out, out_rev) = ctx.saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dy.stride(-1) != 1:
            dy = dy.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
                x, conv1d_weight, conv1d_bias, None, None, None, True
            )
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                              "d (b l) -> b d l", l = L)
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz = selective_scan_cuda.bwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, dy, scan_intermediates, out, dz,
            ctx.delta_softplus,
            False  # option to recompute y
        )
        
        # Reverse part
        dz_rev = torch.empty_like(dz)
        dconv1d_out_rev, ddelta_rev, dA_rev, dB_rev, dC_rev, dD_rev, ddelta_bias_rev, dz_rev = selective_scan_cuda.bwd(
            conv1d_out.flip([-1]), delta.flip([-1]), A_rev, B.flip([-1]), C.flip([-1]), D, z.flip([-1]), delta_bias, dy.flip([-1]), scan_intermediates_rev, out_rev, dz_rev,
            ctx.delta_softplus,
            False  # option to recompute y
        )
        
        dconv1d_out = dconv1d_out + dconv1d_out_rev.flip([-1])
        ddelta = ddelta + ddelta_rev.flip([-1])
        dB = dB + dB_rev.flip([-1])
        dC = dC + dC_rev.flip([-1])
        dD = dD + dD_rev
        ddelta_bias = ddelta_bias + ddelta_bias_rev
        dz = dz + dz_rev.flip([-1])
        
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
                dA, dA_rev, dB, dC, dD,
                ddelta_bias if delta_bias is not None else None,
                dB_proj_bias, dC_proj_bias, None)


def bimamba_v1_inner_fn(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    A, A_rev, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    return BiMambaV1InnerFn.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                              A, A_rev, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus)
