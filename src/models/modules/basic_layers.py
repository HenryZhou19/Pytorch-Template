from torch import nn


class MLP(nn.Module):

    def __init__(self, in_channel: int, out_channels: list, activate_layer=nn.GELU, drop=0.0) -> None:
        super().__init__()
        self.layers = nn.Sequential()
        for idx, out_channel in enumerate(out_channels):
            self.layers.append(nn.Linear(in_channel, out_channel))
            if idx < len(out_channels) - 1:
                self.layers.append(activate_layer())
            self.layers.append(nn.Dropout(drop))
            in_channel = out_channel
        self.out_channel = out_channels[-1]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
