from torch import nn
from torch.nn import functional as F

from .modules.model_base import ModelBase, model_register
from .modules.simple_net import SimpleNet
from .modules.unet import UNetXd


@model_register('simple')
class SimpleModel(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.model.backbone == 'default':
            self.backbone = SimpleNet()
        else:
            raise NotImplementedError(f'backbone "{cfg.model.backbone}" has not been implemented yet for {self.__class__}.')
        
        self.head = nn.Sequential(
            nn.Linear(100, 1),
            nn.Sigmoid()
            )
        
        self.set_no_weight_decay_by_param_names([
            'head.0.weight'
        ])
        self.set_no_reinit_by_param_names([
            'head.0.bias'
        ])

        self._custom_init_all(self._fn_vanilla_custom_init)

    def forward(self, inputs: dict) -> dict:
        x = inputs['x']
        # x = self._grad_checkpoint(self.backbone, x)
        x = self.backbone(x)
        x = self.head(x)
        return {
            'pred_y': x
        }


@model_register('simple_unet2d')
class SimpleUNet2DModel(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.model.backbone == 'default':
            self.backbone = UNetXd(in_channels=3, layer_out_channels=[64, 128, 256, 512, 1024], dimension=2)
        else:
            raise NotImplementedError(f'backbone "{cfg.model.backbone}" has not been implemented yet for {self.__class__}.')
        
    def forward(self, inputs: dict) -> dict:
        x = inputs['x']
        x = self.backbone(x)
        return {
            'pred_y': x
        }
 
 
@model_register('simple_unet3d')
class SimpleUNet3DModel(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.model.backbone == 'default':
            self.backbone = UNetXd(in_channels=3, dimension=3)
        else:
            raise NotImplementedError(f'backbone "{cfg.model.backbone}" has not been implemented yet for {self.__class__}.')

    def forward(self, inputs: dict) -> dict:
        x = inputs['x']
        x = self.backbone(x)
        return {
            'pred_y': x
        }
        

@model_register('lenet')
class LeNet(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert cfg.model.backbone == 'default'
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, inputs: dict) -> dict:
        x = inputs['x']
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return {
            'pred_scores': x
        }
        

@model_register('lenet_multi_optimizer')
class LeNetMultiOptimizer(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert cfg.model.backbone == 'default'
        conv1 = nn.Conv2d(1, 6, kernel_size=5)
        conv2 = nn.Conv2d(6, 16, kernel_size=5)
        fc1 = nn.Linear(16 * 4 * 4, 120)
        fc2 = nn.Linear(120, 84)
        fc3 = nn.Linear(84, 10)
        
        self.convs = nn.ModuleList([conv1, conv2])
        self.fcs = nn.ModuleList([fc1, fc2, fc3])

    def forward(self, inputs: dict) -> dict:
        x = inputs['x']
        x = F.relu(self.convs[0](x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.convs[1](x))
        x_conv_out = F.max_pool2d(x, 2)
        x = x_conv_out.view(-1, 16 * 4 * 4).detach()
        x = F.relu(self.fcs[0](x))
        x = F.relu(self.fcs[1](x))
        x = self.fcs[2](x)
        return {
            'conv_out': x_conv_out,
            'pred_scores': x
        }
        

class LeNetConvs(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        return x
    
    
class LeNetFCs(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        

@model_register('lenet_multi_optimizer_v2')
class LeNetMultiOptimizerV2(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert cfg.model.backbone == 'default'       
        self.convs = LeNetConvs()
        self.fcs = LeNetFCs()

    def forward(self, inputs: dict) -> dict:
        x = inputs['x']
        x_conv_out = self.convs(x)
        x = x_conv_out.view(-1, 16 * 4 * 4).detach()
        x = self.fcs(x)
        return {
            'conv_out': x_conv_out,
            'pred_scores': x
        }