import math
from typing import Literal

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
    
    
    class SimpleWarmUpCosineAnnealingScheduler:
        '''
        If need a scheduler that can be used without the optimizer, use this class.
        
        Special case: A scheduler without cosine annealing (only with the warmup phase)
            can be achieved by setting ``T_warmup == T_max`` when ``dataloader == None``.
            or just setting ``warmup_epochs == epochs`` when ``dataloader`` is provided.
            Warning: In the end, ``T_warmup" is actually set to ``T_max - 1`` to make it valid.
        '''
        def __init__(
            self,
            base_value,
            min_value,
            dataloader=None,
            epochs=None,
            warmup_epochs=0,
            T_max=None,
            T_warmup=0,
            warmup_fn: Literal["no_warmup", "constant", "linear", 'exponential', 'cosine']='linear',
            current_index=-1,
            ):
            
            if dataloader is not None:
                assert epochs is not None, 'epochs should be provided if dataloader is provided.'
                print('The dataloader is provided, "T_max" and "T_warmup" will be calculated based on the dataloader with "epochs" and "warmup_epochs".')
                      
                len_dataloader = len(dataloader)
                T_max = epochs * len_dataloader
                T_warmup = warmup_epochs * len_dataloader
                if warmup_epochs == epochs:
                    T_warmup = T_max - 1
                    print('The warmup_epochs is equal to epochs, so the T_warmup will be set to T_max - 1, making it a warmup only scheduler.')
            else:
                assert T_max is not None, 'T_max should be provided if dataloader is not provided.'
                if T_warmup == T_max:
                    T_warmup = T_max - 1
                    print('The T_warmup is equal to T_max, so the T_warmup will be set to T_max - 1, making it a warmup only scheduler.')
            
            assert 0 <= T_warmup < T_max, 'T_warmup should be in the range of [0, T_max).'
            self.base_value = base_value
            self.min_value = min_value
            self.T_max = T_max
            self.T_warmup = T_warmup
            self.warmup_fn = WarmUpFn.get_warmup_fn(warmup_fn)
            self.current_index = current_index
        
        def reset_index(self, index=-1):
            self.current_index = index
        
        def _get_value(self, specific_index=None):
            if specific_index is not None:
                index = specific_index
            else:
                index = self.current_index
            assert 0 <= index < self.T_max, 'Index out of range.'
            if index < self.T_warmup:
                alpha = self.warmup_fn(index, self.T_warmup)
            else:
                alpha = float(index - self.T_warmup) / (self.T_max - self.T_warmup)
                alpha = 0.5 + 0.5 * math.cos(math.pi * alpha)
            return self.min_value + alpha * (self.base_value - self.min_value)
        
        def next(self):
            self.current_index += 1
            return self._get_value()
        
        def get_all_as_list(self):
            return [self._get_value(i) for i in range(self.T_max)]
        
        def __call__(self):
            return self.next()