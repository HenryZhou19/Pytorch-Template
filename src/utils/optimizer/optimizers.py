from typing import List

import torch

from src.models.modules.model_base import ModelBase
from src.utils.misc import LoggerMisc, ModelMisc

from .modules.arbitrary_scheduler import BasicScheduler
from .schedulers import SchedulerUtils


class IntegratedOptimizer:
    def __init__(
        self,
        root_module_name: str,
        root_module: ModelBase,
        optimizer: torch.optim.Optimizer,
        max_grad_norm_group_list: list,
        lr_scale_scheduler: BasicScheduler,
        wd_scale_scheduler: BasicScheduler,
        scaler: torch.amp.GradScaler,
        lr_default: float,
        wd_default: float,
        ):
        self.root_module_name = root_module_name
        self.root_module= root_module
        
        self.optimizer = optimizer
        self.max_grad_norm_group_list = max_grad_norm_group_list
        self.scaler = scaler
        self.lr_scale_scheduler = lr_scale_scheduler
        self.wd_scale_scheduler = wd_scale_scheduler
        
        self.lr_default_base = lr_default
        self.wd_default_base = wd_default
        self.lr_default = None
        self.wd_default = None
        
        self._init_schedulers()
        
    def state_dict(self):
        state_dict = {
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict() if self.scaler is not None else None,
            'lr_scale_scheduler': self.lr_scale_scheduler.state_dict(),
            'wd_scale_scheduler': self.wd_scale_scheduler.state_dict(),
            }
        return state_dict
    
    def load_state_dict(self, state_dict, skip_optimizer=False, skip_scaler=False, skip_scheduler=False):
        if not skip_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        if not skip_scaler and self.scaler is not None:
            assert 'scaler' in state_dict, 'the checkpoint does not contain "scaler".'
            self.scaler.load_state_dict(state_dict['scaler'])
        if not skip_scheduler:
            self.lr_scale_scheduler.load_state_dict(state_dict['lr_scale_scheduler'])
            self.wd_scale_scheduler.load_state_dict(state_dict['wd_scale_scheduler'])
        
    def optimize(self, fn_before_step=None):
        grad_norm_value_list, grad_norm_name_list = [], []
        need_unscaled_optimizer = self.scaler is not None
        
        for max_grad_norm_group in self.max_grad_norm_group_list:
            name, params, value = max_grad_norm_group['name'], max_grad_norm_group['params'], max_grad_norm_group['value']
            if value > 0:  # only clip when max_grad_norm > 0, otherwise skip; max_grad_norm = .inf can be used to disable clipping while logging the grad norm
                if need_unscaled_optimizer:
                    self.scaler.unscale_(self.optimizer)
                    need_unscaled_optimizer = False
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters=params, max_norm=value)
                grad_norm_name_list.append(name)
                grad_norm_value_list.append(grad_norm)
        
        if fn_before_step is not None:
            fn_before_step()
        
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.step()
            
        self.zero_grad()
        return grad_norm_value_list, grad_norm_name_list
    
    ## the following methods are inherited from the inner Optimizer
    @property
    def param_groups(self):
        return self.optimizer.param_groups
    
    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none)
    
    def step(self, closure=None):
        self.optimizer.step(closure)
        
    def _init_schedulers(self):
        self.lr_scale_scheduler.reset_index()
        self.wd_scale_scheduler.reset_index()
        self.schedulers_step()
        
    def schedulers_step(self):
        lr_scale = self.lr_scale_scheduler.step()
        wd_scale = self.wd_scale_scheduler.step()
        for param_group in self.param_groups:
            param_group['lr'] = lr_scale * param_group['lr_base']
            param_group['weight_decay'] = wd_scale * param_group['wd_base']
        self.lr_default = lr_scale * self.lr_default_base
        self.wd_default = wd_scale * self.wd_default_base
    

class OptimizerUtils: 
    @staticmethod
    def _print_optimizer_param_groups(optimizer_param_group_list, param_to_name, optimizer_name, loggers, name_total_width=60):
        print(f'\n[{optimizer_name}] parameter groups:', file=loggers.log_file)
        n_groups = len(optimizer_param_group_list)
        for i, group in enumerate(optimizer_param_group_list, start=1):
            other_keys_str = ', '.join([f'{k}={v}' for k, v in group.items() if k not in ['name', 'params']])
            print(f'\tParameter group [{i}/{n_groups}]: "{group.get("name", "N/A")}" ({other_keys_str})', file=loggers.log_file)
            for p in group['params']:
                n_str = f'\t\t{param_to_name.get(id(p), "<NOT_FOUND>")}'
                dash_count = max(name_total_width - len(n_str), 3)
                dash_str = '-' * dash_count
                shape_str = f'shape {list(p.shape)}'
                print(f'{n_str} {dash_str} {shape_str}', file=loggers.log_file)
        print('\n', file=loggers.log_file)
        loggers.log_file.flush()
        
    @staticmethod
    def _print_max_grad_norm_groups(max_grad_norm_group_list, param_to_name, optimizer_name, loggers, name_total_width=60):
        print(f'\n[{optimizer_name}] max grad norm groups:', file=loggers.log_file)
        n_groups = len(max_grad_norm_group_list)
        for i, group in enumerate(max_grad_norm_group_list, start=1):
            other_keys_str = ', '.join([f'{k}={v}' for k, v in group.items() if k not in ['name', 'params']])
            print(f'\tMax grad norm group [{i}/{n_groups}]: "{group.get("name", "N/A")}" ({other_keys_str})', file=loggers.log_file)
            for p in group['params']:
                n_str = f'\t\t{param_to_name.get(id(p), "<NOT_FOUND>")}'
                dash_count = max(name_total_width - len(n_str), 3)
                dash_str = '-' * dash_count
                shape_str = f'shape {list(p.shape)}'
                print(f'{n_str} {dash_str} {shape_str}', file=loggers.log_file)
        print('\n', file=loggers.log_file)
        loggers.log_file.flush()
    
    @staticmethod
    def _get_integrated_optimizer(cfg, optimizer_cfg, train_loader, root_module: ModelBase, root_module_name: str, optimizer_name: str, loggers) -> IntegratedOptimizer:
        
        ## call these specific methods of ModelBase to configure freezing and param groups
        root_module.configure_freezing(
            freeze_module_names=optimizer_cfg.freeze_modules,
            freeze_param_names=optimizer_cfg.freeze_params,
            )
        optimizer_param_group_list = root_module.configure_optimizer_param_groups(
            lr_default=optimizer_cfg.lr_default,
            wd_default=optimizer_cfg.wd_default,
            param_group_rules_cfg=optimizer_cfg.param_group_rules,
            )
        max_grad_norm_group_list = root_module.configure_max_grad_norm_groups(
            max_grad_norm_name_list=optimizer_cfg.max_grad_norm_name,
            max_grad_norm_value_list=optimizer_cfg.max_grad_norm_value,
            max_grad_norm_modules_list=optimizer_cfg.max_grad_norm_modules,
            )
        
        if cfg.info.print_param_groups:
            param_to_name = {id(p): n for n, p in root_module.named_parameters()}
            OptimizerUtils._print_optimizer_param_groups(optimizer_param_group_list, param_to_name, optimizer_name, loggers)
            OptimizerUtils._print_max_grad_norm_groups(max_grad_norm_group_list, param_to_name, optimizer_name, loggers)
        
        if optimizer_cfg.optimizer_choice == 'adamw':
            optimizer = torch.optim.AdamW(optimizer_param_group_list, eps=getattr(optimizer_cfg, 'adamw_eps', 1.0e-8))
        elif optimizer_cfg.optimizer_choice == 'sgd':
            optimizer = torch.optim.SGD(optimizer_param_group_list, momentum=getattr(optimizer_cfg, 'sgd_momentum', 0))
        else:
            raise ValueError(f'Unknown optimizer choice: {optimizer_cfg.optimizer_choice}')
        
        # scaler
        if cfg.amp.amp_enabled and cfg.amp.amp_mode == 'fp16':
            print('\nUsing torch.amp.GradScaler for automatic mixed precision (AMP) training with fp16.\n')
            scaler = torch.amp.GradScaler(device='cuda', enabled=True)
        else:
            if not cfg.amp.amp_enabled:
                print(f'\nNot using GradScaler as amp_enabled is set to False (maybe caused by CPU training).\n')
            else:
                print(f'\nNot using GradScaler as amp_enabled is set to True and amp_mode is set to "{cfg.amp.amp_mode}" (not "fp16").\n')
            scaler = None
        
        # prepare for lr and wd scale schedulers
        lr_scale_scheduler, wd_scale_scheduler = SchedulerUtils.get_lr_wd_scale_schedulers(
            cfg=cfg,
            cfg_for_optimizer=optimizer_cfg,
            train_loader=train_loader,
            )
        
        integrated_optimizer = IntegratedOptimizer(
            root_module_name=root_module_name,
            root_module=root_module,
            optimizer=optimizer,
            max_grad_norm_group_list=max_grad_norm_group_list,
            lr_scale_scheduler=lr_scale_scheduler,
            wd_scale_scheduler=wd_scale_scheduler,
            scaler=scaler,
            lr_default=optimizer_cfg.lr_default,
            wd_default=optimizer_cfg.wd_default,
            )
        
        return integrated_optimizer
    
    @staticmethod
    def get_integrated_optimizers(cfg, model_without_ddp: ModelBase, train_loader, loggers) -> List[IntegratedOptimizer]:
        integrated_optimizers = []
        
        for optimizer_name in cfg.trainer.all_optimizer_names:
            optimizer_cfg = getattr(cfg.trainer, optimizer_name)
            
            if optimizer_name != 'optimizer':
                assert hasattr(optimizer_cfg, 'identifier'), f'Non-main optimizer config `{optimizer_name}` must have an identifier.'
                root_module_name = optimizer_cfg.identifier
                root_module = getattr(model_without_ddp, root_module_name)
            else:
                assert len(cfg.trainer.all_optimizer_names) == 1, 'Main optimizer config must be the only one.'
                root_module_name = 'main'
                root_module = model_without_ddp
            assert isinstance(root_module, ModelBase), f'Root module for optimizer `{optimizer_name}` must be a ModelBase instance.'
            
            integrated_optimizer = OptimizerUtils._get_integrated_optimizer(
                cfg=cfg,
                optimizer_cfg=optimizer_cfg,
                train_loader=train_loader,
                root_module=root_module,
                root_module_name=root_module_name,
                optimizer_name=optimizer_name,
                loggers=loggers,
                )
            integrated_optimizers.append(integrated_optimizer)
        
        return integrated_optimizers
