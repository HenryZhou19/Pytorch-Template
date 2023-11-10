import math

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_channel: int, out_channels: list, activate_layer: nn.Module=nn.GELU, dropout=0.0) -> None:
        super().__init__()
        self.mlp = nn.Sequential()
        for idx, out_channel in enumerate(out_channels):
            self.mlp.append(nn.Linear(in_channel, out_channel))
            if idx < len(out_channels) - 1:
                self.mlp.append(activate_layer())
            self.mlp.append(nn.Dropout(dropout))
            in_channel = out_channel
        self.out_channel = out_channels[-1]

    def forward(self, x):
        x = self.mlp(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length, dropout=0.1):
        super().__init__()
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, d_model)  # [L, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1 for N, L, d_model]
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]  # [N, L, d_model]
        return self.dropout(x)