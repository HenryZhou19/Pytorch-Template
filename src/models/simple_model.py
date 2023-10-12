from torch import nn

from src.models.modules.simple_net import SimpleNet


class SimpleModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.model.backbone == 'default':
            self.model = SimpleNet()
        else:
            raise NotImplementedError(f'backbone "{cfg.model.backbone}" has not been implemented yet for {self.__class__}.')

    def forward(self, x):
        x = self.model(x)
        return {
            'pred_y': x
        }

