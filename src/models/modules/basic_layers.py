import math

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_channel: int, out_channels: list, activation_layer: nn.Module=nn.GELU, dropout=0.0, final_activation=False) -> None:
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
        
    def forward(self, x):
        x = self.mlp(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length, type='sinusoidal', dropout=0.1):
        super().__init__()
        if type == 'sinusoidal': 
            position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_seq_length, d_model)  # [L, d_model]
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1(N), L, d_model]
            self.register_buffer('pe', pe)
        elif type == 'learnable':
            self.pe = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # [N, L, d_model]  # [N, L, d_model]
        return self.dropout(x)


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
