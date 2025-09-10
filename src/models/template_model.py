from typing import List

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
        # x = ModelMisc.grad_checkpoint(self.training, self.backbone, x)
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
    
    def check_special_param_group_rules(self, default_param_dict, param_group_rules_cfg):
        
        conv_lr_scale = param_group_rules_cfg.conv_lr_scale
        conv_wd_scale = param_group_rules_cfg.conv_wd_scale
        
        if 'conv' in default_param_dict['name']:
            default_param_dict['lr_base'] = default_param_dict['lr_base'] * conv_lr_scale
            default_param_dict['wd_base'] = default_param_dict['wd_base'] * conv_wd_scale
            
        if 'conv1.weight' in default_param_dict['name']:
            default_param_dict['logging'] = True
        if 'fc1.weight' in default_param_dict['name']:
            default_param_dict['logging'] = True
            
        return default_param_dict

    def forward(self, inputs: dict) -> dict:
        x = inputs['x']
        x = F.silu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.silu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 4 * 4)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = self.fc3(x)
        return {
            'pred_scores': x
        }
        

class LeNetConvs(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.silu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        return x
    
    
class LeNetFCs(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = self.fc3(x)
        return x
        

@model_register('lenet_multi_optimizer')
class LeNetMultiOptimizerV2(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert cfg.model.backbone == 'default'       
        self.convs = LeNetConvs(cfg)
        self.fcs = LeNetFCs(cfg)

    def forward(self, inputs: dict) -> dict:
        x = inputs['x']
        x_conv_out = self.convs(x)
        x = x_conv_out.view(-1, 16 * 4 * 4).detach()
        x = self.fcs(x)
        return {
            'conv_out': x_conv_out,
            'pred_scores': x
        }