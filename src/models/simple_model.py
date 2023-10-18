from .modules.model_base import ModelBase, register
from .modules.simple_net import SimpleNet


@register('simple')
class SimpleModel(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.model.backbone == 'default':
            self.model = SimpleNet()
        else:
            raise NotImplementedError(f'backbone "{cfg.model.backbone}" has not been implemented yet for {self.__class__}.')
        
        # self._freeze_layers(['0', 'bias'], verbose=True)

    def forward(self, **inputs):
        x = inputs['x']
        x = self.model(x)
        return {
            'pred_y': x
        }