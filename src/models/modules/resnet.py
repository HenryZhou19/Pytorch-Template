import torch.nn as nn
from torch.nn import functional as F


class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class ResNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        output = F.relu(self.bn1(output))

        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(extra_x + output)
    
    
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        self.layer1 = nn.Sequential(
            ResNetBasicBlock(64, 64, 1),
            ResNetBasicBlock(64, 64, 1)
            )

        self.layer2 = nn.Sequential(
            ResNetDownBlock(64, 128, [2, 1]),
            ResNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(
            ResNetDownBlock(128, 256, [2, 1]),
            ResNetBasicBlock(256, 256, 1)
            )

        self.layer4 = nn.Sequential(
            ResNetDownBlock(256, 512, [2, 1]),
            ResNetBasicBlock(512, 512, 1)
            )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.final_linear = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.final_linear(x.reshape(x.shape[0], -1))
        return x


class ResNetLight(nn.Module):
    def __init__(self, in_channels=3, out_channels=[32, 64, 128], out=nn.AdaptiveAvgPool2d):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels[0]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        
        self.layer1 = nn.Sequential(
            ResNetBasicBlock(out_channels[0], out_channels[0], 1),
            ResNetBasicBlock(out_channels[0], out_channels[0], 1),
            )
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                ResNetDownBlock(out_channels[i], out_channels[i + 1], [2, 1]),
                ResNetBasicBlock(out_channels[i + 1], out_channels[i + 1], 1)
                ) for i in range(len(out_channels) - 1)
        ])
        self.out = out(output_size=(1, 1))  # nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.Identity

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.out(x)
        return x
