from torch import nn

from src.models.modules.simple_net import SimpleNet


class SimpleModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.model.backbone == 'simple':
            self.model = SimpleNet()
        else:
            raise NotImplementedError(f'backbone "{cfg.model.backbone}" has not been implemented yet.')

    def forward(self, x):
        return self.model(x)

