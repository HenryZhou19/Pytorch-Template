from torch import nn

from .modules.model_register import register
from .modules.simple_net import SimpleNet


@register('simple')
class SimpleModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.model.backbone == 'default':
            self.model = SimpleNet()
        else:
            raise NotImplementedError(f'backbone "{cfg.model.backbone}" has not been implemented yet for {self.__class__}.')

    def forward(self, **inputs):
        x = inputs['x']
        x = self.model(x)
        return {
            'pred_y': x
        }