from functools import partial

import torch.utils.checkpoint as checkpoint
from torch import nn

from src.utils.misc import LoggerMisc
from src.utils.register import Register

model_register = Register('model')

class ModelBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.do_grad_checkpoint = cfg.trainer.grad_checkpoint
                    
    def _grad_checkpoint(self, func, *args, **kwargs):
        if self.do_grad_checkpoint and self.training:
            return checkpoint.checkpoint(func, *args, use_reentrant=False, **kwargs)
        else:
            return func(*args, **kwargs)
    
    @staticmethod
    def _fn_custom_init(module, **kwargs):
        raise NotImplementedError
            
    @staticmethod
    def _fn_vanilla_custom_init(module, std=0.02):
        if isinstance(module, nn.modules.conv._ConvNd):
            nn.init.normal_(module.weight, 0.0, std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, std)
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
            
    def _custom_init_all(self, custom_init_fn=None, **kwargs):
        if custom_init_fn is None:
            custom_init_fn = self._fn_custom_init
        print(LoggerMisc.block_wrapper(
            f'Customize the initialization of all params in {self.__class__.__name__}...\n' + 
            f'with no_reinit_list:\n{LoggerMisc.list_to_multiline_string(self.no_reinit_list)}'
            ))
        self.apply(partial(custom_init_fn, **kwargs))
    
    @property
    def no_weight_decay_list(self):
        '''
        all param.ndim <= 1 or name.endswith(".bias") are not decayed by default
        the list of params' names that are not decayed can be customized by overriding this property
        
        the name here must match the name in model_without_ddp.named_parameters()
        '''
        return []
    
    @property
    def no_reinit_list(self):
        '''
        make sure the params in this list are not re-initialized when calling _custom_init_all() or any custom init methods
        
        the name here must match the name in model_without_ddp.named_parameters()
        '''
        return []
        
    def forward(self, inputs: dict) -> dict:
        raise NotImplementedError
        