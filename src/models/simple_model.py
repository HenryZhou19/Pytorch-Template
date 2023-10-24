from torch import nn

from .modules.model_base import ModelBase, register
from .modules.simple_net import SimpleNet
from .modules.unet import UNetXd


@register('simple')
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
        
        # self._freeze_layers(['backbone'], verbose=True)

    def forward(self, **inputs):
        x = inputs['x']
        x = self._grad_checkpoint(self.backbone, x)
        x = self.head(x)
        return {
            'pred_y': x
        }


@register('simple_unet2d')
class SimpleUNet2DModel(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.model.backbone == 'default':
            self.backbone = UNetXd(in_channels=3, layer_out_channels=[64, 128, 256, 512, 1024], dimension=2)
        else:
            raise NotImplementedError(f'backbone "{cfg.model.backbone}" has not been implemented yet for {self.__class__}.')
        
        # self._freeze_layers(['backbone'], verbose=True)

    def forward(self, **inputs):
        x = inputs['x']
        x = self.backbone(x)
        return {
            'pred_y': x
        }
 
 
@register('simple_unet3d')
class SimpleUNet3DModel(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.model.backbone == 'default':
            self.backbone = UNetXd(in_channels=3, dimension=3)
        else:
            raise NotImplementedError(f'backbone "{cfg.model.backbone}" has not been implemented yet for {self.__class__}.')
        
        # self._freeze_layers(['backbone'], verbose=True)

    def forward(self, **inputs):
        x = inputs['x']
        x = self.backbone(x)
        return {
            'pred_y': x
        }