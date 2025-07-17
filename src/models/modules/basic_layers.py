import math

import torch
from torch import nn

from .basic_functions import (adapt_conv_2d_load_from_state_dict,
                              adapt_conv_3d_load_from_state_dict,
                              adapt_L_C_parameter_load_from_state_dict,
                              trunc_normal_init_linear_weights)


class MLP(nn.Module):
    def __init__(self, in_channel: int, out_channels: list, activation_layer: nn.Module=nn.SiLU, dropout=0.0, final_activation=False, trunc_normal_init=False) -> None:
        super().__init__()
        self.mlp = nn.Sequential()
        for idx, out_channel in enumerate(out_channels):
            self.mlp.append(nn.Linear(in_channel, out_channel))
            if idx < len(out_channels) - 1 or final_activation:
                self.mlp.append(activation_layer())
            if dropout > 0.0:
                self.mlp.append(nn.Dropout(dropout))
            in_channel = out_channel
        self.out_channel = out_channels[-1]
        
        if trunc_normal_init:
            self.apply(trunc_normal_init_linear_weights)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DSMLP(nn.Module):
    def __init__(self, in_channel: int, out_channels: int, inter_channels: int, activation_layer: nn.Module=nn.SiLU, trunc_normal_init=False) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_channel, inter_channels)
        self.w2 = nn.Linear(inter_channels, out_channels)
        self.w3 = nn.Linear(in_channel, inter_channels)
        self.activation = activation_layer()
        
        if trunc_normal_init:
            self.apply(trunc_normal_init_linear_weights)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.activation(self.w1(x)) * self.w3(x))


class SoftMoE(nn.Module):
    def __init__(
        self,
        input_dim,
        num_experts=4,
        noisy_gating=True,  # noisy gating (improves load balancing)
        gate_noise_std=1.0,  # noise std for noisy gating
        aux_loss_coef=0.01,  # coefficient of load-balance loss
        external_experts=True,  # use external experts (if False, use inner default expert)
        hidden_dim=None,  # hidden dimension of experts (only used if external_experts=False)
        output_dim=None,  # output dimension of experts (only used if external_experts=False)
        external_gate=True,  # external gate (if None, use inner default gate)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.noisy_gating = noisy_gating
        self.gate_noise_std = gate_noise_std
        self.aux_loss_coef = aux_loss_coef
        self.external_experts = external_experts
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.external_gate = external_gate
        
        # Experts
        if self.external_experts:
            self.experts = None
        else:
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
            ])
        
        # Gate
        if self.external_gate:
            self.gate = None
        else:
            self.gate = nn.Linear(input_dim, num_experts)

    def noisy_gates(self, x, gate_logits):
        if self.external_gate:
            assert gate_logits is not None, "When using external gate, gate_logits must be provided."
        else:
            gate_logits = self.gate(x)
        if self.training and self.noisy_gating:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * self.gate_noise_std
        return gate_logits

    def forward(self, x, expert_outputs: list=None, gate_logits: torch.Tensor=None):
        """
        input: [..., D]
        output: [..., output_dim]
        """
        shape = x.shape[:-1]
        gate_logits = self.noisy_gates(x, gate_logits)                           # [..., num_experts]
        gate_weights = nn.functional.softmax(gate_logits, dim=-1)               # [..., num_experts]
        
        
        if self.external_experts:
            assert expert_outputs is not None, "When using external experts, expert_outputs must be provided."
        else:
            # Compute each expert output
            expert_outputs = []
            for expert in self.experts:
                expert_outputs.append(expert(x))         # [..., output_dim]
        
        expert_outputs = torch.stack(expert_outputs, dim=-2)          # [..., num_experts, output_dim]
        moe_output = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=-2)  # [..., output_dim]

        # load balance loss（soft)
        if not self.training:
            return moe_output, None
        
        load = gate_weights.mean(dim=tuple(range(len(shape))))      # (num_experts,)
        moe_aux_loss = (load * load).sum() * self.aux_loss_coef

        return moe_output, moe_aux_loss


class PositionalEncoding(nn.Module):
    def __init__(self, d_pos, max_seq_length, type='learnable', init_magnitute=1.0, d_start_idx=0, drop_out=0.0):
        super().__init__()
        if type is None:
            type = 'none'
            
        if type == 'learnable':
            self.pe = nn.Parameter(torch.randn(max_seq_length, d_pos) * init_magnitute)  # [L, d_model]
            self.magnitude = None
        elif type == 'sinusoidal': 
            self._init_sinusoidal_pe(d_pos, max_seq_length)  # [L, d_model]
            self.magnitude = None
        elif type == 'scalable_sinusoidal':
            self._init_sinusoidal_pe(d_pos, max_seq_length)  # [L, d_model]
            self.magnitude = nn.Parameter(torch.ones(1, d_pos) * init_magnitute)  # [1, d_model]
        elif type == 'none':
            pe = torch.zeros(max_seq_length, d_pos)  # [L, d_model]
            self.register_buffer('pe', pe, persistent=False)
            self.magnitude = None
        
        self.pos_type = type
        self.d_pos = d_pos
        self.max_seq_length = max_seq_length
        self.d_start_idx = d_start_idx
        assert self.d_start_idx >= 0, f'd_start_idx={d_start_idx} should be >= 0'
        self.d_end_idx = d_start_idx + d_pos
        if drop_out > 0.0:
            self.dropout = nn.Dropout(drop_out)
        else:    
            self.dropout = nn.Identity()
            
    def _init_sinusoidal_pe(self, d_pos, max_seq_length):
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_pos, 2).float() * (-math.log(10000.0) / d_pos))
        pe = torch.zeros(max_seq_length, d_pos)  # [L, d_pos]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)
                
    def _get_pe(self):
        if self.magnitude is not None:
            return self.pe * self.magnitude
        else:
            return self.pe
    
    def expand_pe(self, xx: torch.Tensor, seq_dim):
        seq_length = xx.shape[seq_dim]        
        assert seq_length <= self.max_seq_length, f'x.shape[{seq_dim}]={xx.shape[seq_dim]} > max_seq_length={self.max_seq_length}'
        
        pe = self._get_pe()[:seq_length]  # [L, d_pos]
        shape = [1] * len(xx.shape)
        shape[seq_dim] = seq_length
        shape[-1] = self.d_pos
        return pe.reshape(shape).to(xx.dtype)
        
    def forward(self, x: torch.Tensor, seq_dim):  #  x: [..., L, ..., d_model]
        assert self.d_end_idx <= x.shape[-1], f'd_end_idx={self.d_end_idx} should be <= x.shape[-1]={x.shape[-1]}'
        xx = x[..., self.d_start_idx:self.d_end_idx]  # [..., L, ..., d_pos]
        x[..., self.d_start_idx:self.d_end_idx] = xx + self.dropout(self.expand_pe(xx, seq_dim))
        return x
    
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs,
        ):
        if self.pos_type == 'learnable':
            state_dict = adapt_L_C_parameter_load_from_state_dict(state_dict, prefix + "pe", self.pe)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )


class PatchEmbedding2D(nn.Module):
    def __init__(self, tensor_hw=(224, 224), patch_size=(16, 16), stride=(16, 16), in_channels=3, embed_dim=768, norm_layer=nn.Identity, flatten=True, check_input=False):
        super().__init__()
        self.tensor_hw = tensor_hw
        self.patch_size = patch_size
        self.stride = stride
        self.in_channels = in_channels
        
        self.grid_size = (
            (tensor_hw[0] - patch_size[0]) // stride[0] + 1, 
            (tensor_hw[1] - patch_size[1]) // stride[1] + 1,
            )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim)
        
        self.flatten = flatten
        self.check_input = check_input
        
    def forward(self, x: torch.Tensor):
        if self.check_input:
            N, C, H, W = x.shape
            assert H == self.tensor_hw[0] and W == self.tensor_hw[1], f'Predefined 2D-image size {self.tensor_hw} does not match input 2D-image size {x.shape[2:]}'
            assert C == self.in_channels, f'Input channel {C} does not match predefined channel {self.in_channels}'
        x = self.proj(x)  # [N, C, H, W] -> [N, embed_dim, grid_size[0], grid_size[1]]
        
        x = x.permute(0, 2, 3, 1)  # [N, grid_size[0], grid_size[1], embed_dim]
        if self.flatten:
            x = x.flatten(1, -2)  # [N, grid_size[0], grid_size[1], embed_dim] -> [N, num_patches, embed_dim]
        x = self.norm(x)
        return x
    
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs,
        ):
        key = prefix + "proj.weight"
        state_dict = adapt_conv_2d_load_from_state_dict(state_dict, key, self.proj.weight)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
            )


class PatchEmbedding3D(nn.Module):
    def __init__(self, tensor_dhw=(32, 224, 224), patch_size=(2, 16, 16), stride=(2, 16, 16), in_channels=3, embed_dim=768, norm_layer=nn.Identity, flatten=True, check_input=False):
        super().__init__()
        self.tensor_dhw = tensor_dhw
        self.patch_size = patch_size
        self.stride = stride
        self.in_channels = in_channels
        
        self.grid_size = (
            (tensor_dhw[0] - patch_size[0]) // stride[0] + 1,
            (tensor_dhw[1] - patch_size[1]) // stride[1] + 1,
            (tensor_dhw[2] - patch_size[2]) // stride[2] + 1,
            )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim)
        
        self.flatten = flatten
        self.check_input = check_input
        
    def forward(self, x: torch.Tensor):
        if self.check_input:
            N, C, D, H, W = x.shape
            assert D == self.tensor_dhw[0] and H == self.tensor_dhw[1] and W == self.tensor_dhw[2], f'Predefined 3D-image size {self.tensor_dhw} does not match input 3D-image size {x.shape[2:]}'
            assert C == self.in_channels, f'Input channel {C} does not match predefined channel {self.in_channels}'
        x = self.proj(x)  # [N, C, D, H, W] -> [N, embed_dim, grid_size[0], grid_size[1], grid_size[2]]
        
        x = x.permute(0, 2, 3, 4, 1)  # [N, grid_size[0], grid_size[1], grid_size[2], embed_dim]
        if self.flatten:
            x = x.flatten(1, -2)  # [N, grid_size[0], grid_size[1], grid_size[2], embed_dim] -> [N, num_patches, embed_dim]
        x = self.norm(x)
        return x

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs,
        ):
        state_dict = adapt_conv_3d_load_from_state_dict(state_dict, prefix + "proj.", self.proj)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
            )