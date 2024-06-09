import torch

from src.models.modules.model_base import ModelBase


class OptimizerUtils:
    @staticmethod
    def _get_param_dicts_with_specific_lr_wd(cfg, model_without_ddp: ModelBase):
        
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
                    'lr': cfg.trainer.optimizer.lr_default,
                    'weight_decay': cfg.trainer.optimizer.wd_default,
                },
                'default_no_wd': {
                    'params': [],
                    'lr': cfg.trainer.optimizer.lr_default,
                    'weight_decay': 0.,
                }
            }
            if hasattr(cfg.trainer.optimizer, 'param_groups'):
                for key, value in vars(cfg.trainer.optimizer.param_groups).items():
                    assert key.startswith(lr_mark) or key.startswith(wd_mark), f'Unknown param_groups config: {key}'
                    if key.startswith(lr_mark):
                        param_group_name = key[len(lr_mark):]
                        param_group_name_list.append(param_group_name)
                        param_groups[param_group_name] = {
                            'params': [],
                            'lr': value,
                            'weight_decay': getattr(cfg.trainer.optimizer.param_groups, key.replace(lr_mark, wd_mark)),
                            }
                        param_groups[param_group_name + '_no_wd'] = {
                            'params': [],
                            'lr': value,
                            'weight_decay': 0.,
                            }
            return param_groups, param_group_name_list
        
        param_groups, param_group_name_list = create_all_param_groups()
        
        for param_name, param in model_without_ddp.named_parameters():
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
    def get_optimizer(cfg, model_without_ddp) -> tuple[torch.optim.Optimizer, torch.cuda.amp.GradScaler]:
        param_dicts_with_lr_wd = OptimizerUtils._get_param_dicts_with_specific_lr_wd(cfg, model_without_ddp)
        if cfg.trainer.optimizer.optimizer_choice == 'adamw':
            optimizer = torch.optim.AdamW(param_dicts_with_lr_wd)
        elif cfg.trainer.optimizer.optimizer_choice == 'sgd':
            optimizer = torch.optim.SGD(param_dicts_with_lr_wd, momentum=cfg.trainer.optimizer.sgd_momentum)
        else:
            raise ValueError(f'Unknown optimizer choice: {cfg.trainer.optimizer.optimizer_choice}')
        
        if cfg.env.amp.amp_enabled:
            if cfg.env.amp.amp_mode == 'fp16':
                scaler = torch.cuda.amp.GradScaler(enabled=True)
                setattr(scaler, 'custom_dtype', torch.float16)
            elif cfg.env.amp.amp_mode == 'bf16':
                scaler = torch.cuda.amp.GradScaler(enabled=False)
                setattr(scaler, 'custom_dtype', torch.bfloat16)
            else:
                raise ValueError(f'Unknown amp.amp_mode: {cfg.env.amp.amp_mode}')
        else:
            scaler = None
        
        return optimizer, scaler
