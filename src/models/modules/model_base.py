from functools import partial

import torch.utils.checkpoint as checkpoint
from torch import nn

from src.utils.misc import LoggerMisc, ModelMisc
from src.utils.register import Register

model_register = Register('model')

class ModelBase(nn.Module):
    registered_name: str
    
    def __init__(self, cfg):
        super().__init__()
        self.ema_mode = False
        self.infer_mode = False
        self.cfg = cfg
        self.no_state_modules = dict()
        
        self.custom_inited = False
        
    def set_ema_mode(self, ema_mode):
        self.ema_mode = ema_mode
    
    def set_infer_mode(self, infer_mode):
        self.infer_mode = infer_mode
        
    def print_states(self, prefix='', force=True):
        print(f'{prefix}Model --- training mode: {self.training}, infer_mode: {self.infer_mode}, ema_mode: {self.ema_mode}', force=force)
        
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for module in self.no_state_modules.values():
            module.to(*args, **kwargs)
        return self
    
    @staticmethod
    def _fn_custom_init(module, **kwargs):
        raise NotImplementedError
            
    @staticmethod
    def _fn_vanilla_custom_init(module, std=0.02):
        if isinstance(module, nn.modules.conv._ConvNd):
            if ModelMisc._re_init_check(module, 'weight'):
                nn.init.normal_(module.weight, 0.0, std)
            if module.bias is not None and ModelMisc._re_init_check(module, 'bias'):
                    nn.init.constant_(module.bias, 0)
            
        elif isinstance(module, nn.Linear):
            if ModelMisc._re_init_check(module, 'weight'):
                nn.init.normal_(module.weight, 0.0, std)
            if module.bias is not None and ModelMisc._re_init_check(module, 'bias'):
                nn.init.constant_(module.bias, 0)
            
        elif isinstance(module, nn.Embedding):
            if ModelMisc._re_init_check(module, 'weight'):
                nn.init.normal_(module.weight, 0.0, std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            if ModelMisc._re_init_check(module, 'weight'):
                nn.init.constant_(module.weight, 1)
            if ModelMisc._re_init_check(module, 'bias'):
                nn.init.constant_(module.bias, 0)
            
        elif isinstance(module, nn.modules.normalization.GroupNorm):
            if ModelMisc._re_init_check(module, 'weight'):
                nn.init.constant_(module.weight, 1)
            if ModelMisc._re_init_check(module, 'bias'):
                nn.init.constant_(module.bias, 0)
            
    def _custom_init_all(self, custom_init_fn=None, **kwargs):
        if custom_init_fn is None:
            custom_init_fn = self._fn_custom_init
        print(LoggerMisc.block_wrapper(f'Customize the initialization of all params in {self.__class__.__name__}...'))
        self.apply(partial(custom_init_fn, **kwargs))
        print('Custom initialization done.\n')
        self.custom_inited = True
    
    def set_no_weight_decay_by_param_names(self, param_names_list):
        '''
        all param.ndim <= 1 or name.endswith(".bias") are not decayed by default
        the list of params' names that are not decayed can be customized by overriding this property
        
        the name here must match the name in model_without_ddp.named_parameters()
        '''
        for name, param in self.named_parameters():
            if name in param_names_list:
                setattr(param, '_no_weight_decay', True)
    
    def set_no_reinit_by_param_names(self, param_names_list):
        '''
        make sure the params in this list are not re-initialized when calling _custom_init_all() or any custom init methods
        
        the name here must match the name in model_without_ddp.named_parameters()
        '''
        assert not self.custom_inited, 'Cannot set no_reinit after custom init'
        for name, param in self.named_parameters():
            if name in param_names_list:
                setattr(param, '_no_reinit', True)
        
    def forward(self, inputs: dict) -> dict:
        raise NotImplementedError
        
    def before_one_epoch(self, *args, **kwargs):
        pass
    
    def before_all_epochs(self, *args, **kwargs):
        pass