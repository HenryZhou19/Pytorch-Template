import torch


class OptimizerUtils:
    @staticmethod
    def _get_param_dicts_with_specific_lr_wd(cfg, model_without_ddp: torch.nn.Module):
        
        def match_param_groups(param_name, param_group_names):
            flag = False
            param_group_name = 'default'
            for lr_group_name in param_group_names:
                if lr_group_name in param_name:
                    assert not flag, f'Name {param_name} matches multiple param_group names in {param_group_names}.'
                    flag = True
                    param_group_name = lr_group_name
            return param_group_name

        if hasattr(cfg.trainer.optimizer, 'param_groups'):
            lr_mark = 'lr_'
            wd_mark = 'wd_'
            param_groups = {
                'default': {
                    'params': [],
                    'lr': cfg.trainer.optimizer.lr_default,
                    'weight_decay': cfg.trainer.optimizer.wd_default,
                }
            }
            for k, v in vars(cfg.trainer.optimizer.param_groups).items():
                assert k.startswith(lr_mark) or k.startswith(wd_mark), f'Unknown param_groups config: {k}'
                if k.startswith(lr_mark): 
                    param_groups[k[len(lr_mark):]] = {
                        'params': [],
                        'lr': v,
                        'weight_decay': getattr(cfg.trainer.optimizer.param_groups, k.replace(lr_mark, wd_mark)),
                        }
            for n, p in model_without_ddp.named_parameters():
                if p.requires_grad:
                    param_groups[match_param_groups(n, param_groups.keys())]['params'].append(p)

            param_dicts_with_lr_wd = []
            for k, v in param_groups.items():
                param_dicts_with_lr_wd.append({
                    **v,
                    'group_name': k
                    })
        else:  # if no cfg.optimizer.param_groups, then all params use 'default' config
            param_dicts_with_lr_wd = [{
                'params': [p for _, p in model_without_ddp.named_parameters()
                            if p.requires_grad],
                'lr': cfg.trainer.optimizer.lr_default,
                'weight_decay': cfg.trainer.optimizer.wd_default,
                'group_name': 'default'
                }]
        
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
        
        scaler = torch.cuda.amp.GradScaler() if cfg.env.amp and cfg.env.device=='cuda' else None
        
        return optimizer, scaler
