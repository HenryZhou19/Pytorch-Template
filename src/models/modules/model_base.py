from functools import partial
from typing import List

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
        
        self.freeze_modules = {}  # will be set in configure_param_groups_and_freezing()
        self.freeze_params = {}  # will be set in configure_param_groups_and_freezing()
        
    def set_ema_mode(self, ema_mode):
        self.ema_mode = ema_mode
        for module in self.children():
            if hasattr(module, 'set_ema_mode'):
                module.set_ema_mode(ema_mode)
    
    def set_infer_mode(self, infer_mode):
        self.infer_mode = infer_mode
        for module in self.children():
            if hasattr(module, 'set_infer_mode'):
                module.set_infer_mode(infer_mode)
        
    def print_states(self, prefix='', force=True):
        print(f'{prefix}{self.__class__.__name__} --- training mode: {self.training}, infer_mode: {self.infer_mode}, ema_mode: {self.ema_mode}', force=force)
        
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
    
    def configure_freezing(self, freeze_module_names: List[str], freeze_param_names: List[str]):
        '''
        called in the initialization of the trainer (when creating optimizers)
            configure freezing rules and do the freezing before configuring param groups and creating optimizers
        return: None
        '''
        self.freeze_modules = ModelMisc.get_specific_submodules_with_full_names(
            self,
            freeze_module_names,
            strict=False,
            )
        self.freeze_params = ModelMisc.get_specific_params_with_full_names(
            self,
            freeze_param_names,
            strict=False,
            )
        if len(self.freeze_modules) > 0 or len(self.freeze_params) > 0:
            print(LoggerMisc.block_wrapper('Freezing modules before creating optimizers...\n\tThese params will not be added to any param groups.\n\tSo if you want to train them later, do not freeze them here.\n'), '>')
        ModelMisc.unfreeze_or_freeze_modules(
            modules_dict=self.freeze_modules,
            is_trainable=False,
            verbose=True,
            )
        ModelMisc.unfreeze_or_freeze_params(
            params_dict=self.freeze_params,
            is_trainable=False,
            verbose=True,
            )
    
    def configure_optimizer_param_groups(self, lr_default, wd_default, param_group_rules_cfg):
        '''
        called in the initialization of the trainer (when creating optimizers)
            configure param groups after doing freezing and before creating optimizers
        return: optimizer_param_group_list for the corresponding optimizer
        '''
        default_param_groups = {
            'default': {
                'params': [],
                'lr_base': lr_default,
                'wd_base': wd_default,
            },
            'default_no_wd': {
                'params': [],
                'lr_base': lr_default,
                'wd_base': 0.,
            }
        }
        no_wd_names = param_group_rules_cfg.no_wd_names
        no_wd_max_ndim = param_group_rules_cfg.no_wd_max_ndim
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            no_decay = False
            if param.ndim <= no_wd_max_ndim:
                no_decay = True
            if any([no_wd_name in name for no_wd_name in no_wd_names]):
                no_decay = True
            if getattr(param, '_no_weight_decay', False):
                no_decay = True
            if no_decay:
                default_param_groups['default_no_wd']['params'].append(param)
            else:
                default_param_groups['default']['params'].append(param)
        
        optimizer_param_group_list = []
        for k, v in default_param_groups.items():
            optimizer_param_group_list.append({
                'group_name': k,
                **v,
                })  
        optimizer_param_group_list = [g for g in optimizer_param_group_list if len(g['params']) > 0]
        
        return optimizer_param_group_list
    
    def configure_max_grad_norm_groups(self, max_grad_norm_name_list: List[str], max_grad_norm_value_list: List[float], max_grad_norm_modules_list: List[List[str]]):
        max_grad_norm_group_dict = {}
        
        for max_grad_norm_name, max_grad_norm_value, max_grad_norm_modules in zip(max_grad_norm_name_list, max_grad_norm_value_list, max_grad_norm_modules_list):
            assert max_grad_norm_name not in max_grad_norm_group_dict, f'Duplicate max_grad_norm_name: {max_grad_norm_name}'
            assert max_grad_norm_value >= 0, f'max_grad_norm_value must be non-negative, got {max_grad_norm_value}'
            
            if max_grad_norm_modules == ['']:
                modules = self
            else:
                found_modules = ModelMisc.get_specific_submodules_with_full_names(self, max_grad_norm_modules)
                modules = nn.ModuleList([module for module in found_modules.values()])
            max_grad_norm_params = []
            for param in modules.parameters():
                if not param.requires_grad:
                    continue
                max_grad_norm_params.append(param)
            max_grad_norm_group_dict[max_grad_norm_name] = {
                'params': max_grad_norm_params,
                'value': max_grad_norm_value,
            }
        
        max_grad_norm_group_list = []
        for k, v in max_grad_norm_group_dict.items():
            max_grad_norm_group_list.append({
                'group_name': k,
                **v,
                })  
        max_grad_norm_group_list = [g for g in max_grad_norm_group_list if len(g['params']) > 0]
        
        return max_grad_norm_group_list
        
    def before_one_epoch(self, *args, **kwargs):
        '''
        this will be called in the trainer right before calling self.train()
        '''
        pass
    
    def before_all_epochs(self, *args, **kwargs):
        '''
        this will be called in the trainer before show_model_info, resume training or loading pretrained weights
        '''
    
    def train(self, mode = True):
        '''
        this will be called in the trainer right after self.before_one_epoch()
        '''
        super().train(mode)
        if mode:  # only call freeze when set to train mode (as .eval() will also call train(False))
            ModelMisc.train_or_eval_modules(
                modules_dict=self.freeze_modules,
                is_train=False,
                verbose=False,
                )
        
    def eval(self):
        '''
        this will be called in the trainer or tester before validation or testing
        '''
        super().eval()