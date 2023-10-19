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
                    
    def grad_checkpoint(self, func, *args, **kwargs):
        if self.do_grad_checkpoint and self.training:
            return checkpoint.checkpoint(func, *args, use_reentrant=False, **kwargs)
        else:
            return func(*args, **kwargs)
        
    def forward(self, **inputs):
        raise NotImplementedError
        