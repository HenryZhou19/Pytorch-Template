import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dimension, activation_layer=nn.SiLU, norm='batch', padding_mode='zeros', res_in_block=True,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        assert dimension in [2, 3], 'Unsupported dimension'
        ConvXd = nn.Conv2d if dimension == 2 else nn.Conv3d
        if norm == 'batch':
            NormXd = nn.BatchNorm2d if dimension == 2 else nn.BatchNorm3d
        elif norm == 'instance':
            NormXd = nn.InstanceNorm2d if dimension == 2 else nn.InstanceNorm3d
        else:
            # XXX: if using other norm modules, bias may be needed in ConvXd
            raise NotImplementedError('Unsupported norm type')
        self.conv_block = nn.Sequential(
            ConvXd(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, padding_mode=padding_mode),
            NormXd(out_channels),
            activation_layer(inplace=True),
            ConvXd(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, padding_mode=padding_mode),
            NormXd(out_channels),
            )
        self.conv_final_activation = activation_layer(inplace=True)
        self.res_in_block = res_in_block
        if self.res_in_block:
            self.conv_res = nn.Sequential(
                ConvXd(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                NormXd(out_channels),
                )

    def forward(self, x):
        x = self.conv_final_activation(self.conv_res(x) + self.conv_block(x)) if self.res_in_block else self.conv_final_activation(self.conv_block(x))
        return x


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, dimension, padding_mode='zeros', no_down_dim=None, res_in_block=True):
        super().__init__()
        assert dimension in [2, 3], 'Unsupported dimension'
        MaxPoolXd = nn.MaxPool2d if dimension == 2 else nn.MaxPool3d
        kernel_size = [2] * dimension
        if no_down_dim is not None:
            if isinstance(no_down_dim, int):
                no_down_dim = (no_down_dim, )
            for d in no_down_dim:
                kernel_size[d - 2] = 1
        kernel_size = tuple(kernel_size)  
        stride = kernel_size
        self.unet_down = nn.Sequential(
            MaxPoolXd(kernel_size=kernel_size, stride=stride),
            ConvBlock(in_channels, out_channels, dimension, padding_mode=padding_mode, res_in_block=res_in_block)
        )

    def forward(self, x):
        return self.unet_down(x)


class UpSampling(nn.Module):
    def __init__(self, in_channels, cat_channels, out_channels, dimension, use_conv_transpose, padding_mode='zeros', no_up_dim=None, res_in_block=True):
        super().__init__()
        assert dimension in [2, 3], 'Unsupported dimension'
        ConvTransposeXd = nn.ConvTranspose2d if dimension == 2 else nn.ConvTranspose3d
        kernel_size = [2] * dimension
        if no_up_dim is not None:
            if isinstance(no_up_dim, int):
                no_up_dim = (no_up_dim, )
            for d in no_up_dim:
                kernel_size[d - 2] = 1
        kernel_size = tuple(kernel_size)  
        stride = kernel_size
        if use_conv_transpose:
            self.up = ConvTransposeXd(in_channels, in_channels, kernel_size=kernel_size, stride=stride)
        else:
            self.up = nn.Upsample(scale_factor=kernel_size, mode='bilinear' if dimension == 2 else 'trilinear', align_corners=True)
        self.conv = ConvBlock(in_channels + cat_channels, out_channels, dimension, padding_mode=padding_mode, res_in_block=res_in_block)

    def forward(self, x, cat_features=None):
        x = self.up(x)
        if cat_features is not None:
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
    def __init__(self, in_channels, layer_out_channels=[64, 128, 256, 512], final_out_channels=None, dimension=2, use_conv_transpose=False, padding_mode='zeros', res_in_block=True, features_out=False):
        super().__init__()
        assert dimension in [2, 3], 'Unsupported dimension'
        if final_out_channels is None:
            final_out_channels = in_channels

        self.in_conv = ConvBlock(in_channels, layer_out_channels[0], dimension, padding_mode=padding_mode, res_in_block=res_in_block)
        self.down_layers = nn.ModuleList([
            DownSampling(layer_out_channels[i], layer_out_channels[i + 1], dimension, padding_mode, res_in_block=res_in_block) for i in range(len(layer_out_channels) - 1)
        ])
        self.up_layers = nn.ModuleList([
            UpSampling(layer_out_channels[i], layer_out_channels[i - 1], layer_out_channels[i - 1], dimension, use_conv_transpose, padding_mode, res_in_block=res_in_block) for i in range(len(layer_out_channels) - 1, 0, -1)
        ])
        self.out_conv = LastConv(layer_out_channels[0], final_out_channels, dimension)
        
        self.features_out = features_out

    def forward(self, x):
        # down
        down_out_list = []
        down_out_list.append(self.in_conv(x))
        for down_layer in self.down_layers:
            down_out_list.append(down_layer(down_out_list[-1]))

        # up
        if self.features_out:
            feature_list = []
            x = down_out_list.pop()
            feature_list.append(x)
            for up_layer in self.up_layers:
                x = up_layer(x, down_out_list.pop())
                feature_list.append(x)
            x = self.out_conv(x)
            return x, feature_list
        else:
            x = down_out_list.pop()
            for up_layer in self.up_layers:
                x = up_layer(x, down_out_list.pop())
            x = self.out_conv(x)
            return x


class TimeUpscaleUNet3d(nn.Module):
    def __init__(self, in_channels, up_scale=4, layer_out_channels=[64, 128, 256], final_out_channels=None, use_conv_transpose=False, padding_mode='zeros', res_in_block=True):
        super().__init__()
        assert 2 ** (len(layer_out_channels) - 1) == up_scale, 'up_scale must be 2 ** (len(layer_out_channels) - 1)'
        kernel_size = (2, 1, 1)
        stride = kernel_size
        
        if final_out_channels is None:
            final_out_channels = in_channels

        self.in_conv = ConvBlock(in_channels, layer_out_channels[0], 3, padding_mode=padding_mode, res_in_block=res_in_block)
        self.down_layers = nn.ModuleList([
            DownSampling(layer_out_channels[i], layer_out_channels[i + 1], 3, padding_mode, no_down_dim=2, res_in_block=res_in_block) for i in range(len(layer_out_channels) - 1)
        ])
        self.special_up = nn.ModuleList([
            nn.ConvTranspose3d(layer_out_channels[i],
                            layer_out_channels[i],
                            kernel_size=tuple((torch.tensor(kernel_size) ** (len(layer_out_channels) - 1 - i)).tolist()),
                            stride=tuple((torch.tensor(stride) ** (len(layer_out_channels) - 1 - i)).tolist())
                            ) for i in range(len(layer_out_channels) - 1)
        ])
        self.up_layers = nn.ModuleList([
            UpSampling(layer_out_channels[i], layer_out_channels[i - 1], layer_out_channels[i - 1], 3, use_conv_transpose, padding_mode, res_in_block=res_in_block) for i in range(len(layer_out_channels) - 1, 0, -1)
        ])
        self.out_conv = LastConv(layer_out_channels[0], final_out_channels, 3)

    def forward(self, x):  # [N, 3, L_fused, h, w] -> [N, 3, L_all, h ,w]
        # down
        cat_list = []
        x = self.in_conv(x)  # [N, c1, L_fused, h, w]
        cat_list.append(self.special_up[0](x))
        for idx, down_layer in enumerate(self.down_layers):
            x = down_layer(x)
            if idx < len(self.down_layers) - 1:  # [N, c2, L_fused, h/2, w/2]
                cat_list.append(self.special_up[idx + 1](x))

        # up
        for up_layer in self.up_layers:
            x = up_layer(x, cat_list.pop())
        x = self.out_conv(x)
        
        return x
    

class TimeDownscaleUNet3d(nn.Module):
    def __init__(self, in_channels, down_scale=4, layer_out_channels=[64, 128, 256], final_out_channels=None, use_conv_transpose=False, padding_mode='zeros', res_in_block=True):
        super().__init__()
        assert 2 ** (len(layer_out_channels) - 1) == down_scale, 'down_scale must be 2 ** (len(layer_out_channels) - 1)'
        kernel_size = (2, 1, 1)
        stride = kernel_size
        
        if final_out_channels is None:
            final_out_channels = in_channels

        self.in_conv = ConvBlock(in_channels, layer_out_channels[0], 3, padding_mode=padding_mode, res_in_block=res_in_block)
        self.down_layers = nn.ModuleList([
            DownSampling(layer_out_channels[i], layer_out_channels[i + 1], 3, padding_mode, res_in_block=res_in_block) for i in range(len(layer_out_channels) - 1)
        ])
        self.special_down = nn.ModuleList([
            nn.MaxPool3d(
                kernel_size=tuple((torch.tensor(kernel_size) ** (len(layer_out_channels) - 1 - i)).tolist()),
                stride=tuple((torch.tensor(stride) ** (len(layer_out_channels) - 1 - i)).tolist())) for i in range(len(layer_out_channels) - 1)
        ])
        self.up_layers = nn.ModuleList([
            UpSampling(layer_out_channels[i], layer_out_channels[i - 1], layer_out_channels[i - 1], 3, use_conv_transpose, padding_mode, no_up_dim=2, res_in_block=res_in_block) for i in range(len(layer_out_channels) - 1, 0, -1)
        ])
        self.out_conv = LastConv(layer_out_channels[0], final_out_channels, 3)

    def forward(self, x):  # [N, 3, L_all, h, w] -> [N, 3, L_fused, h ,w]
        # down
        cat_list = []
        x = self.in_conv(x)  # [N, c1, L_fused, h, w]
        cat_list.append(self.special_down[0](x))
        for idx, down_layer in enumerate(self.down_layers):
            x = down_layer(x)
            if idx < len(self.down_layers) - 1:  # [N, c2, L_fused, h/2, w/2]
                cat_list.append(self.special_down[idx + 1](x))
                
        # up
        for up_layer in self.up_layers:
            x = up_layer(x, cat_list.pop())
        x = self.out_conv(x)
        
        return x