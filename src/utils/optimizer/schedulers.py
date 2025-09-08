import torch

from .modules.warmup_scheduler import *


class SchedulerUtils:
    @staticmethod
    def get_warmup_lr_scheduler(cfg, cfg_for_optimizer, optimizer, scaler, train_loader) -> torch.optim.lr_scheduler._LRScheduler:
        
        len_train_loader = len(train_loader)
        T_max = cfg.trainer.epochs * len_train_loader
        if cfg_for_optimizer.scheduler.warmup_steps >= 0:
            T_warmup = cfg_for_optimizer.scheduler.warmup_steps
        elif cfg_for_optimizer.scheduler.warmup_epochs >= 0:
            T_warmup = cfg_for_optimizer.scheduler.warmup_epochs * len_train_loader
            if cfg_for_optimizer.scheduler.warmup_epochs == cfg.trainer.epochs:
                T_warmup = T_max - 1
                print('The warmup_epochs is equal to epochs, so the T_warmup will be set to T_max - 1, making it a warmup only scheduler.')
        else:
            T_warmup = 0
            
        kwargs = {
            'optimizer': optimizer,
            'scaler': scaler,
            'do_grad_accumulation': cfg.trainer.grad_accumulation > 1,
            'T_max': T_max,
            'T_warmup': T_warmup,
            'warmup_fn': WarmUpFn.get_warmup_fn(cfg_for_optimizer.scheduler.warmup_type),
            'lr_min_factor': cfg_for_optimizer.scheduler.lr_min_factor,
        }
        if cfg_for_optimizer.scheduler.scheduler_choice == 'vanilla':
            return WarmUpVanillaLR(**kwargs) 
        elif cfg_for_optimizer.scheduler.scheduler_choice == 'cosine':
            return WarmUpCosineAnnealingLR(**kwargs)
        elif cfg_for_optimizer.scheduler.scheduler_choice == 'cosine_restart':
            kwargs.pop('T_max')  # T_max is not used in CosineAnnealingRestartLR
            if cfg_for_optimizer.scheduler.lr_first_cycle_steps is not None:
                first_cycle_steps = cfg_for_optimizer.scheduler.lr_first_cycle_steps
            elif cfg_for_optimizer.scheduler.lr_first_cycle_epochs is not None:
                first_cycle_steps = len_train_loader * cfg_for_optimizer.scheduler.lr_first_cycle_epochs
            else:
                raise ValueError('lr_first_cycle_steps and lr_first_cycle_epochs cannot be both None.')
            kwargs.update({
                'first_cycle_steps': first_cycle_steps,
                'cycle_mult': getattr(cfg_for_optimizer.scheduler, 'lr_cycle_mult', 1.0),
                'gamma': getattr(cfg_for_optimizer.scheduler, 'lr_cycle_gamma', 1.0),
            })
            return WarmupCosineAnnealingRestartLR(**kwargs)
        elif cfg_for_optimizer.scheduler.scheduler_choice == 'cosine_multi_cycle':
            kwargs.pop('T_max')  # T_max is not used in WarmupCosineAnnealingMultiCycleLR
            if cfg_for_optimizer.scheduler.lr_cycle_epochs_list is not None:
                cycle_steps_list = [len_train_loader * lr_cycle_epoch for lr_cycle_epoch in cfg_for_optimizer.scheduler.lr_cycle_epochs_list]
            else:
                raise ValueError('lr_cycle_epochs_list cannot be None.')
            kwargs.update({
                'cycle_steps_list': cycle_steps_list,
                'gamma': getattr(cfg_for_optimizer.scheduler, 'lr_cycle_gamma', 1.0),
            })
            return WarmupCosineAnnealingMultiCycleLR(**kwargs)
        elif cfg_for_optimizer.scheduler.scheduler_choice == 'linear':
            return WarmUpLinearLR(**kwargs)
        elif cfg_for_optimizer.scheduler.scheduler_choice == 'multistep':
            if cfg_for_optimizer.scheduler.lr_milestones_steps is not None:
                step_milestones = cfg_for_optimizer.scheduler.lr_milestones_steps
            elif cfg_for_optimizer.scheduler.lr_milestones_epochs is not None:
                step_milestones = [len_train_loader * lr_milestones_epoch for lr_milestones_epoch in cfg_for_optimizer.scheduler.lr_milestones_epochs]
            else:
                raise ValueError('lr_milestones_steps and lr_milestones_epochs cannot be both None.')
            kwargs.update({
                'step_milestones': step_milestones,
                'gamma': cfg_for_optimizer.scheduler.lr_decay_gamma,
            })
            return WarmUpMultiStepLR(**kwargs)
        else:
            raise ValueError(f'Unknown scheduler choice: {cfg_for_optimizer.scheduler.scheduler_choice}')
        
    @staticmethod
    def get_warmup_wd_scale_scheduler(cfg, cfg_for_optimizer, train_loader) -> BasicScheduler:
        '''
        A scheduler that only scales the weight decay during the warmup phase.
        Currently, only a cosine warmup from start_value to end_value is supported.
        Note: This is a scaler only, which will multiply the basic weight decay value in the IntegratedOptimizer.
        '''
        len_train_loader = len(train_loader)
        T_max = cfg.trainer.epochs * len_train_loader
        wd_start_scale = getattr(cfg_for_optimizer.scheduler, 'wd_start_scale', 1.0)
        wd_end_scale = getattr(cfg_for_optimizer.scheduler, 'wd_end_scale', 1.0)
        
        kwargs = {
            'start_value': wd_start_scale,
            'end_value': wd_end_scale,
            'T_max': T_max,
            'warmup_type': 'cosine',
        }
        return SimpleWarmupScheduler(**kwargs) 
