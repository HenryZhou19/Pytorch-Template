import torch

from .modules.warmup_scheduler import *


class SchedulerUtils:   
    @staticmethod
    def get_warmup_lr_scheduler(cfg, optimizer, scaler, train_loader) -> torch.optim.lr_scheduler._LRScheduler:
        
        len_train_loader = len(train_loader)
        if cfg.trainer.scheduler.warmup_steps >= 0:
            warmup_iters = cfg.trainer.scheduler.warmup_steps
        elif cfg.trainer.scheduler.warmup_epochs >= 0:
            warmup_iters = cfg.trainer.scheduler.warmup_epochs * len_train_loader
        else:
            warmup_iters = 0
            
        kwargs = {
            'optimizer': optimizer,
            'scaler': scaler,
            'do_grad_accumulation': cfg.trainer.grad_accumulation > 1,
            'T_max': cfg.trainer.epochs * len_train_loader,
            'T_warmup': warmup_iters,
            'warmup_factor': cfg.trainer.scheduler.warmup_factor,
            'lr_min': cfg.trainer.scheduler.lr_min,
        }
        if cfg.trainer.scheduler.scheduler_choice == 'vanilla':
            kwargs.pop('lr_min')
            return WarmUpLR(**kwargs) 
        elif cfg.trainer.scheduler.scheduler_choice == 'cosine':
            return WarmUpCosineAnnealingLR(**kwargs)
        elif cfg.trainer.scheduler.scheduler_choice == 'linear':
            return WarmUpLinearLR(**kwargs)
        elif cfg.trainer.scheduler.scheduler_choice == 'multistep':
            if cfg.trainer.scheduler.lr_milestones_steps is not None:
                step_milestones = cfg.trainer.scheduler.lr_milestones_steps
            elif cfg.trainer.scheduler.lr_milestones_epochs is not None:
                step_milestones = [len_train_loader * lr_milestones_epoch for lr_milestones_epoch in cfg.trainer.scheduler.lr_milestones_epochs]
            else:
                raise ValueError('lr_milestones_steps and lr_milestones_epochs cannot be both None.')
            kwargs.update({
                'step_milestones': step_milestones,
                'gamma': cfg.trainer.scheduler.lr_decay_gamma,
            })
            return WarmUpMultiStepLR(**kwargs)
        else:
            raise ValueError(f'Unknown scheduler choice: {cfg.trainer.scheduler.scheduler_choice}')