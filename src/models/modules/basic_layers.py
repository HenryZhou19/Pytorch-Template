import math

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_channel: int, out_channels: list, activation_layer: nn.Module=nn.SiLU, dropout=0.0, final_activation=False) -> None:
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DSMLP(nn.Module):
    def __init__(self, in_channel: int, out_channels: int, inter_channels: int, activation_layer: nn.Module=nn.SiLU) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_channel, inter_channels)
        self.w2 = nn.Linear(inter_channels, out_channels)
        self.w3 = nn.Linear(in_channel, inter_channels)
        self.activation = activation_layer()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.activation(self.w1(x)) * self.w3(x))


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
            self.register_buffer('pe', pe)
            self.magnitude = None
        
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
        self.register_buffer('pe', pe)
                
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
