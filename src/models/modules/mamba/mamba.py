# Mamba Copyright (c) 2023, Tri Dao, Albert Gu.
# Modifed by Jiaheng Zhou

from functools import partial
from typing import Optional, Union

import torch
from torch import nn

from .modules.mamba_block import MambaBlock
from .modules.norm import RMSNorm


class MambaLayer(nn.Module):
    def __init__(
        self,
        layer_index : int,
        mamba_block_cls: MambaBlock,
        norm_cls: Union[nn.LayerNorm, RMSNorm],
        residual_in_fp32=False,
        ):
        '''
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"
        
        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        '''
        super().__init__()
        self.layer_index = layer_index
        self.residual_in_fp32 = residual_in_fp32
        
        self.mamba_block: MambaBlock = mamba_block_cls(layer_index=layer_index)
        self.norm = norm_cls()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        inference_params=None,
        ):
        r"""Pass the input through the encoder layer.
        
        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = mamba_block(norm(residual))
        """
        
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        
        hidden_states = self.mamba_block(hidden_states, inference_params=inference_params)
        return hidden_states, residual
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mamba_block.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class Mamba(nn.Module):
    def __init__(
        self,
        n_layer: int,
        d_model: int,
        mamba_block_cls: MambaBlock,
        mamba_block_config: dict,
        rms_norm=False,
        norm_epsilon=1e-5,
        final_norm=True,
        residual_in_fp32: bool=False,
        device=None,
        dtype=None,
        ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        
        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm,
            normalized_shape=d_model,
            eps=norm_epsilon,
            **factory_kwargs,
            )
        mamba_block_cls = partial(
            mamba_block_cls,
            d_model=d_model,
            **mamba_block_config,
            **factory_kwargs,
            )
        
        self.mamba_layers = nn.ModuleList(
            [
                MambaLayer(
                    layer_index=layer_idx,
                    mamba_block_cls=mamba_block_cls,
                    norm_cls=norm_cls,
                    residual_in_fp32=residual_in_fp32,
                )
                for layer_idx in range(n_layer)
            ]
        )
        
        if final_norm:
            self.final_norm: Union[nn.LayerNorm, RMSNorm] = norm_cls()
            
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            idx: mamba_layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for idx, mamba_layer in enumerate(self.mamba_layers)
        }
        
    def forward(self, hidden_states: torch.Tensor, inference_params=None) -> torch.Tensor:
        residual = None
        for layer in self.mamba_layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.final_norm(residual.to(dtype=self.final_norm.weight.dtype))
        
        return hidden_states
