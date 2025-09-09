from typing import List

import torch

from src.utils.misc import LoggerMisc, ModelMisc

from .modules.arbitrary_scheduler import BasicScheduler
from .schedulers import SchedulerUtils


class IntegratedOptimizer:
    def __init__(self, root_module_name, optimizer, lr_scale_scheduler, wd_scale_scheduler, scaler, root_module, max_grad_norm_dict, freeze_modules, freeze_params):
        self.root_module_name: str = root_module_name
        
        self.optimizer: torch.optim.Optimizer = optimizer
        self.scaler: torch.amp.GradScaler = scaler
        self.lr_scale_scheduler: BasicScheduler = lr_scale_scheduler
        self.wd_scale_scheduler: BasicScheduler = wd_scale_scheduler
        
        self.root_module: torch.nn.Module = root_module
        self.freeze_modules: list = freeze_modules
        self.freeze_params: list = freeze_params
        
        self.max_grad_norm_name_list = list(max_grad_norm_dict.keys())
        self.max_grad_norm_params_list = [list(v['params']) for v in max_grad_norm_dict.values()]
        self.max_grad_norm_value_list = [v['value'] for v in max_grad_norm_dict.values()]
        
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
        if self.scaler is not None:
            already_unscaled_optimizer = False
            for name, params, value in zip(self.max_grad_norm_name_list, self.max_grad_norm_params_list, self.max_grad_norm_value_list):
                if value > 0:  # only clip when max_grad_norm > 0, otherwise skip; max_grad_norm = .inf can be used to disable clipping while logging the grad norm
                    if not already_unscaled_optimizer:
                        already_unscaled_optimizer = True
                        self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(parameters=params, max_norm=value)
                    grad_norm_name_list.append(name)
                    grad_norm_value_list.append(grad_norm)
            if fn_before_step is not None:
                fn_before_step()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            for name, params, value in zip(self.max_grad_norm_name_list, self.max_grad_norm_params_list, self.max_grad_norm_value_list):
                if value > 0:  # only clip when max_grad_norm > 0, otherwise skip; max_grad_norm = .inf can be used to disable clipping while logging the grad norm
                    grad_norm = torch.nn.utils.clip_grad_norm_(parameters=params, max_norm=value)
                    grad_norm_name_list.append(name)
                    grad_norm_value_list.append(grad_norm)
            if fn_before_step is not None:
                fn_before_step()
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
        wd_scale = self.wd_scale_scheduler.step()
        lr_scale = self.lr_scale_scheduler.step()
        for param_group in self.param_groups:
            param_group['lr_base'] = param_group['lr']
            param_group['lr'] = lr_scale * param_group['lr_base']
            param_group['weight_decay_base'] = param_group['weight_decay']
            param_group['weight_decay'] = wd_scale * param_group['weight_decay_base'] 
        
    def schedulers_step(self):
        lr_scale = self.lr_scale_scheduler.step()
        wd_scale = self.wd_scale_scheduler.step()
        for param_group in self.param_groups:
            param_group['lr'] = lr_scale * param_group['lr_base']
            param_group['weight_decay'] = wd_scale * param_group['weight_decay_base']
    

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
            if param.ndim <= 1 or param_name.endswith('.bias') or getattr(param, '_no_weight_decay', False) or 'norm' in param_name or 'gamma' in param_name or 'beta' in param_name:
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
    def _get_max_grad_norm_params_with_value(max_grad_norm_modules, max_grad_norm_value, root_module, root_module_name, optimizer_name):
        assert len(max_grad_norm_modules) > 0, f'max_grad_norm_modules should be a non-empty list (can be [\'\'] for the whole model).'
        if max_grad_norm_modules == ['']:
            print(LoggerMisc.block_wrapper(f'{optimizer_name} for {root_module_name} --- grad norm modules: ALL, with value: {max_grad_norm_value}'))
            max_grad_norm_params = root_module.parameters()
        else:
            found_modules = ModelMisc.get_specific_submodules_with_full_names(root_module, max_grad_norm_modules)
            print(LoggerMisc.block_wrapper(f'{optimizer_name} for {root_module_name} --- grad norm modules: {max_grad_norm_modules}, with value: {max_grad_norm_value}'))
            max_grad_norm_params = torch.nn.ModuleList(
                [module for module in found_modules.values()]
                ).parameters()
        return {
            'params': max_grad_norm_params,
            'value': max_grad_norm_value,
            }
    
    @staticmethod
    def _print_param_groups(param_dicts_with_lr_wd, root_module, optimizer_name, loggers):
        print(f'\n\nOptimizer [{optimizer_name}] parameter groups:', file=loggers.log_file)
        param_to_name = {id(p): n for n, p in root_module.named_parameters()}
        for i, group in enumerate(param_dicts_with_lr_wd):
            print(f'\tParam group {i}: (weight_decay={group.get("weight_decay", "N/A")}, lr={group.get("lr", "N/A")})', file=loggers.log_file)
            for p in group['params']:
                n = param_to_name.get(id(p), '<NOT_FOUND>')
                print(f'\t\t[{i}] {n:60s} | shape {list(p.shape)}', file=loggers.log_file)
        print('\n\n', file=loggers.log_file)
        loggers.log_file.flush()
    
    @staticmethod
    def _get_integrated_optimizer(cfg, optimizer_cfg, train_loader, root_module, root_module_name, optimizer_name, loggers) -> IntegratedOptimizer:
        param_dicts_with_lr_wd = OptimizerUtils._get_param_dicts_with_specific_lr_wd(optimizer_cfg, root_module)
        
        if cfg.info.print_param_groups:
            OptimizerUtils._print_param_groups(param_dicts_with_lr_wd, root_module, optimizer_name, loggers)
        
        if optimizer_cfg.optimizer_choice == 'adamw':
            optimizer = torch.optim.AdamW(param_dicts_with_lr_wd, eps=getattr(optimizer_cfg, 'adamw_eps', 1.0e-8))
        elif optimizer_cfg.optimizer_choice == 'sgd':
            optimizer = torch.optim.SGD(param_dicts_with_lr_wd, momentum=getattr(optimizer_cfg, 'sgd_momentum', 0))
        else:
            raise ValueError(f'Unknown optimizer choice: {optimizer_cfg.optimizer_choice}')
        
        ## the following attributes are use in the trainer
        max_grad_norm_name_list: List[str] = optimizer_cfg.max_grad_norm_name
        max_grad_norm_value_list: List[float] = optimizer_cfg.max_grad_norm_value
        max_grad_norm_modules_list: List[List[str]] = optimizer_cfg.max_grad_norm_modules
        max_grad_norm_dict = {}
        for max_grad_norm_name, max_grad_norm_value, max_grad_norm_modules in zip(max_grad_norm_name_list, max_grad_norm_value_list, max_grad_norm_modules_list):
            assert max_grad_norm_name not in max_grad_norm_dict, f'Duplicate max_grad_norm_name: {max_grad_norm_name}'
            assert max_grad_norm_value >= 0, f'max_grad_norm_value must be non-negative, got {max_grad_norm_value}'
            max_grad_norm_dict[max_grad_norm_name] = OptimizerUtils._get_max_grad_norm_params_with_value(
                max_grad_norm_modules=max_grad_norm_modules,
                max_grad_norm_value=max_grad_norm_value,
                root_module=root_module,
                root_module_name=root_module_name,
                optimizer_name=optimizer_name,
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
        
        # prepare for lr and wd scale schedulers
        lr_scale_scheduler, wd_scale_scheduler = SchedulerUtils.get_lr_wd_scale_schedulers(
            cfg=cfg,
            cfg_for_optimizer=optimizer_cfg,
            train_loader=train_loader,
            )
        
        integrated_optimizer = IntegratedOptimizer(
            root_module_name=root_module_name,
            optimizer=optimizer,
            lr_scale_scheduler=lr_scale_scheduler,
            wd_scale_scheduler=wd_scale_scheduler,
            scaler=scaler,
            root_module=root_module,
            max_grad_norm_dict=max_grad_norm_dict,
            freeze_modules=freeze_modules,
            freeze_params=freeze_params,
            )
        
        return integrated_optimizer
    
    
    @staticmethod
    def get_integrated_optimizers(cfg, model_without_ddp, train_loader, loggers) -> List[IntegratedOptimizer]:
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
