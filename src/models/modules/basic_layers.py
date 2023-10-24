import math

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_channel: int, out_channels: list, activate_layer=nn.GELU, drop=0.0) -> None:
        super().__init__()
        self.mlp = nn.Sequential()
        for idx, out_channel in enumerate(out_channels):
            self.mlp.append(nn.Linear(in_channel, out_channel))
            if idx < len(out_channels) - 1:
                self.mlp.append(activate_layer())
            self.mlp.append(nn.Dropout(drop))
            in_channel = out_channel
        self.out_channel = out_channels[-1]

    def forward(self, x):
        x = self.mlp(x)
        return x
    
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dimension, activate_layer=nn.ReLU, norm='batch'):
        super().__init__()
        assert dimension in [2, 3], "Unsupported dimension"
        ConvXd = nn.Conv2d if dimension == 2 else nn.Conv3d
        if norm == 'batch':
            NormXd = nn.BatchNorm2d if dimension == 2 else nn.BatchNorm3d
        elif norm == 'instance':
            NormXd = nn.InstanceNorm2d if dimension == 2 else nn.InstanceNorm3d
        else:
            raise NotImplementedError("Unsupported norm type")
        self.conv_block = nn.Sequential(
            ConvXd(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            NormXd(out_channels),
            activate_layer(inplace=True),
            ConvXd(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            NormXd(out_channels),
            activate_layer(inplace=True),
            )

    def forward(self, x):
        x = self.conv_block(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, d_model)  # [L, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1 for N, L, d_model]
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]  # [N, L, d_model]
        return self.dropout(x)