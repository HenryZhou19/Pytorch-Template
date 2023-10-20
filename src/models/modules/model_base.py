from functools import partial

import torch.utils.checkpoint as checkpoint
from torch import nn

from src.utils.register import Register

register = Register('model')

class ModelBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.do_grad_checkpoint = cfg.trainer.grad_checkpoint

    def _freeze_layers(self, freeze_keyword_list, verbose=False):
        def match_keywords(param_name):
            for keyword in freeze_keyword_list:  # list
                if keyword in param_name:
                    return True
            return False
        for (name, param) in self.named_parameters():
            if match_keywords(name):
                param.requires_grad = False
            if verbose:
                print(f'param {name} is trainable: {param.requires_grad}, param_shape: {param.shape}')
                    
    def _grad_checkpoint(self, func, *args, **kwargs):
        if self.do_grad_checkpoint and self.training:
            return checkpoint.checkpoint(func, *args, use_reentrant=False, **kwargs)
        else:
            return func(*args, **kwargs)
    
    @staticmethod
    def _custom_init(module, bias=0., std=0.02):
        if isinstance(module, nn.modules.conv._ConvNd):
            nn.init.normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.modules.normalization.GroupNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
            
    def _custom_init_all(self, init_std=0.02):
        self.apply(partial(self._custom_init, init_std=init_std))
        
    def forward(self, **inputs):
        raise NotImplementedError
        