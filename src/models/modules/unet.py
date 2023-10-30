import torch
import torch.nn as nn

from .basic_layers import ConvBlock


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, dimension):
        super().__init__()
        assert dimension in [2, 3], 'Unsupported dimension'
        MaxPoolXd = nn.MaxPool2d if dimension == 2 else nn.MaxPool3d
        
        self.unet_down = nn.Sequential(
            MaxPoolXd(kernel_size=2, stride=2),
            ConvBlock(in_channels, out_channels, dimension)
        )

    def forward(self, x):
        return self.unet_down(x)


class UpSampling(nn.Module):
    def __init__(self, in_channels, cat_channels, out_channels, dimension, use_conv_transpose):
        super().__init__()
        assert dimension in [2, 3], 'Unsupported dimension'
        ConvTransposeXd = nn.ConvTranspose2d if dimension == 2 else nn.ConvTranspose3d
        
        if use_conv_transpose:
            self.up = ConvTransposeXd(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear' if dimension == 2 else 'trilinear', align_corners=True)
        self.conv = ConvBlock(in_channels + cat_channels, out_channels, dimension)

    def forward(self, x, cat_features):
        x = self.up(x)
        x = torch.cat([x, cat_features], dim=1)
        return self.conv(x)


class LastConv(nn.Module):
    def __init__(self, in_channels, out_channels, dimension):
        super().__init__()
        assert dimension in [2, 3], 'Unsupported dimension'
        ConvXd = nn.Conv2d if dimension == 2 else nn.Conv3d
        self.conv = ConvXd(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetXd(nn.Module):
    def __init__(self, in_channels, layer_out_channels=[64, 128, 256, 512], final_out_channels=None, dimension=2, use_conv_transpose=False):
        super().__init__()
        assert dimension in [2, 3], 'Unsupported dimension'
        if final_out_channels is None:
            final_out_channels = in_channels

        self.in_conv = ConvBlock(in_channels, layer_out_channels[0], dimension)
        self.down_layers = nn.ModuleList([
            DownSampling(layer_out_channels[i], layer_out_channels[i + 1], dimension) for i in range(len(layer_out_channels) - 1)
        ])
        self.up_layers = nn.ModuleList([
            UpSampling(layer_out_channels[i], layer_out_channels[i - 1], layer_out_channels[i - 1], dimension, use_conv_transpose) for i in range(len(layer_out_channels) - 1, 0, -1)
        ])
        self.out_conv = LastConv(layer_out_channels[0], final_out_channels, dimension)

    def forward(self, x):
        # down
        down_out_list = []
        down_out_list.append(self.in_conv(x))
        for down_layer in self.down_layers:
            down_out_list.append(down_layer(down_out_list[-1]))

        # up
        x = down_out_list.pop()
        for up_layer in self.up_layers:
            x = up_layer(x, down_out_list.pop())
        x = self.out_conv(x)
        
        return x
