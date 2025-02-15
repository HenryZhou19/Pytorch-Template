from typing import List

import torch

from src.models.modules.model_base import ModelBase
from src.utils.misc import LoggerMisc

from .schedulers import SchedulerUtils


class IntegratedOptimizer:
    def __init__(self, identifier, optimizer, lr_scheduler, scaler, root_module, max_grad_norm, modules_for_grad_norm, freeze_modules, freeze_params):
        self.identifier: str = identifier
        
        self.optimizer: torch.optim.Optimizer = optimizer
        self.scaler: torch.amp.GradScaler = scaler
        self.lr_scheduler: torch.optim.lr_scheduler._LRScheduler = lr_scheduler
        
        self.root_module: torch.nn.Module = root_module
        self.max_grad_norm: float = max_grad_norm
        self.modules_for_grad_norm: torch.nn.Module = modules_for_grad_norm
        self.freeze_modules: list = freeze_modules
        self.freeze_params: list = freeze_params
        
    def state_dict(self):
        state_dict = {
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict() if self.scaler is not None else None,
            'lr_scheduler': self.lr_scheduler.state_dict(),
            }
        return state_dict
    
    def load_state_dict(self, state_dict, skip_optimizer=False, skip_scaler=False, skip_scheduler=False):
        if not skip_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        if not skip_scaler and self.scaler is not None:
            assert 'scaler' in state_dict, 'the checkpoint does not contain "scaler".'
            self.scaler.load_state_dict(state_dict['scaler'])
        if not skip_scheduler:
            self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        
    def optimize(self, fn_before_step=None):
        grad_norm = None
        if self.scaler is not None:
            if self.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters=self.modules_for_grad_norm.parameters(), max_norm=self.max_grad_norm)
            if fn_before_step is not None:
                fn_before_step()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters=self.modules_for_grad_norm.parameters(), max_norm=self.max_grad_norm)
            if fn_before_step is not None:
                fn_before_step()
            self.step()
        self.zero_grad()
        return grad_norm
    
    ## the following methods are inherited from the inner Optimizer
    @property
    def param_groups(self):
        return self.optimizer.param_groups
    
    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none)
    
    def step(self, closure=None):
        self.optimizer.step(closure)
    

class OptimizerUtils:
    @staticmethod
    def _get_param_dicts_with_specific_lr_wd(optimizer_cfg, root_module: torch.nn.Module):
        
        def match_param_group(param_name, param_group_name_list):
            matched_param_group_name = 'default'
            if len(param_group_name_list) == 1:
                return matched_param_group_name
            flag = False
            for param_group_name in param_group_name_list:
                if param_group_name in param_name:
                    assert not flag, f'Name {param_name} matches multiple param_group names in {param_group_name_list}.'
                    flag = True
                    matched_param_group_name = param_group_name
            return matched_param_group_name
        
        def create_all_param_groups():
            lr_mark = 'lr_'
            wd_mark = 'wd_'
            param_group_name_list = ['default']
            param_groups = {
                'default': {
                    'params': [],
                    'lr': optimizer_cfg.lr_default,
                    'weight_decay': optimizer_cfg.wd_default,
                },
                'default_no_wd': {
                    'params': [],
                    'lr': optimizer_cfg.lr_default,
                    'weight_decay': 0.,
                }
            }
            if hasattr(optimizer_cfg, 'param_groups'):
                for key, value in vars(optimizer_cfg.param_groups).items():
                    assert key.startswith(lr_mark) or key.startswith(wd_mark), f'Unknown param_groups config: {key}'
                    if key.startswith(lr_mark):
                        param_group_name = key[len(lr_mark):]
                        param_group_name_list.append(param_group_name)
                        param_groups[param_group_name] = {
                            'params': [],
                            'lr': value,
                            'weight_decay': getattr(optimizer_cfg.param_groups, key.replace(lr_mark, wd_mark)),
                            }
                        param_groups[param_group_name + '_no_wd'] = {
                            'params': [],
                            'lr': value,
                            'weight_decay': 0.,
                            }
            return param_groups, param_group_name_list
        
        param_groups, param_group_name_list = create_all_param_groups()
        
        for param_name, param in root_module.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or param_name.endswith(".bias") or getattr(param, '_no_weight_decay', False):
                param_groups[match_param_group(param_name, param_group_name_list) + '_no_wd']['params'].append(param)
            else:
                param_groups[match_param_group(param_name, param_group_name_list)]['params'].append(param)
        
        # construct the final param_dicts_with_lr_wd
        param_dicts_with_lr_wd = []
        for k, v in param_groups.items():
            param_dicts_with_lr_wd.append({
                'group_name': k,
                **v
                })            
        
        return param_dicts_with_lr_wd
    
    
    @staticmethod
    def _get_modules_for_grad_norm(optimizer_name, root_module, modules_for_grad_norm=None):
        if modules_for_grad_norm is not None:
            print(LoggerMisc.block_wrapper(f'Optimizer {optimizer_name} --- grad norm modules: {modules_for_grad_norm}'))
            modules_for_grad_norm = torch.nn.ModuleList(
                [getattr(root_module, module_name) for module_name in modules_for_grad_norm]
                )
        else:
            print(LoggerMisc.block_wrapper(f'Optimizer {optimizer_name} --- grad norm modules: ALL'))
            modules_for_grad_norm = root_module
        return modules_for_grad_norm
    
    
    @staticmethod
    def _get_integrated_optimizer(cfg, optimizer_cfg, optimizer_identifier, root_module, train_loader) -> IntegratedOptimizer:
        param_dicts_with_lr_wd = OptimizerUtils._get_param_dicts_with_specific_lr_wd(optimizer_cfg, root_module)
        if optimizer_cfg.optimizer_choice == 'adamw':
            optimizer = torch.optim.AdamW(param_dicts_with_lr_wd, eps=getattr(optimizer_cfg, 'adamw_eps', 1.0e-8))
        elif optimizer_cfg.optimizer_choice == 'sgd':
            optimizer = torch.optim.SGD(param_dicts_with_lr_wd, momentum=getattr(optimizer_cfg, 'sgd_momentum', 0))
        else:
            raise ValueError(f'Unknown optimizer choice: {optimizer_cfg.optimizer_choice}')
        
        ## the following attributes are use in the trainer
        # max_grad_norm
        max_grad_norm = optimizer_cfg.max_grad_norm
        
        # modules_for_grad_norm
        modules_for_grad_norm = OptimizerUtils._get_modules_for_grad_norm(
            optimizer_identifier,
            root_module,
            getattr(optimizer_cfg, 'modules_for_grad_norm', None)
            )
        
        # freeze_modules
        freeze_modules = getattr(optimizer_cfg, 'freeze_modules', [])
        
        # freeze_params
        freeze_params = getattr(cfg.trainer, 'freeze_params', [])
        
        # scaler
        if cfg.amp.amp_enabled and cfg.amp.amp_mode == 'fp16':
            scaler = torch.amp.GradScaler(device='cuda', enabled=True)
        else:
            scaler = None
        
        # prepare for lr_scheduler
        lr_scheduler = SchedulerUtils.get_warmup_lr_scheduler(cfg, optimizer_cfg, optimizer, scaler, train_loader)
        
        integrated_optimizer = IntegratedOptimizer(
            identifier=optimizer_identifier,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            root_module=root_module,
            max_grad_norm=max_grad_norm,
            modules_for_grad_norm=modules_for_grad_norm,
            freeze_modules=freeze_modules,
            freeze_params=freeze_params,
            )
        
        return integrated_optimizer
    
    
    @staticmethod
    def get_integrated_optimizers(cfg, model_without_ddp, train_loader) -> List[IntegratedOptimizer]:
        integrated_optimizers = []
        
        for name_optimizer in cfg.trainer.name_optimizers:
            optimizer_cfg = getattr(cfg.trainer, name_optimizer)
            
            if name_optimizer != 'optimizer':
                assert hasattr(optimizer_cfg, 'identifier'), f'Non-main optimizer config `{name_optimizer}` must have an identifier.'
                optimizer_identifier = optimizer_cfg.identifier
                root_module = getattr(model_without_ddp, optimizer_cfg.identifier)
            else:
                assert len(cfg.trainer.name_optimizers) == 1, 'Main optimizer config must be the only one.'
                optimizer_identifier = 'main'
                root_module = model_without_ddp
            
            integrated_optimizer = OptimizerUtils._get_integrated_optimizer(
                cfg=cfg,
                optimizer_cfg=optimizer_cfg,
                optimizer_identifier=optimizer_identifier,
                root_module=root_module,
                train_loader=train_loader,
                )
            integrated_optimizers.append(integrated_optimizer)
        
        
        return integrated_optimizers
