import warnings
from typing import List, Tuple

from .modules.arbitrary_scheduler import ArbitraryScheduler, DummyScheduler


class SchedulerUtils:
    @staticmethod
    def get_dummy_scheduler(dummy_value: float = 0.0) -> DummyScheduler:
        '''
        A dummy scheduler with __getitem__ always returning 0.0
        '''
        return DummyScheduler(dummy_value=dummy_value)
    
    @staticmethod
    def get_simple_warmup_annealing_scheduler(
        start_value: float,
        base_value: float,
        end_value: float,
        T_max: int,
        T_warmup: int,
        warmup_type: str = 'linear',
        annealing_type: str = 'cosine',
        ) -> ArbitraryScheduler:
        return ArbitraryScheduler(
            T_list=[T_warmup, T_max],
            key_value_list=[start_value, base_value, end_value],
            moving_type_list=[warmup_type, annealing_type],
            )
    
    @staticmethod
    def get_arbitrary_scheduler(
        phase_epochs: List[int],
        phase_steps: List[int],
        phase_scales: List[float],
        phase_types: List[str],
        iters_per_epoch: int,
        total_iters: int,
        ) -> ArbitraryScheduler:
        if phase_steps is None:
            if phase_epochs is None:
                raise ValueError('Either phase_epochs or phase_steps should be provided.')
            phase_steps = [epoch * iters_per_epoch for epoch in phase_epochs]
        else:
            if phase_epochs is not None:
                warnings.warn('Both phase_epochs and phase_steps are provided. phase_epochs will be ignored.')
        
        return ArbitraryScheduler(
            T_list=phase_steps + [total_iters],
            key_value_list=phase_scales,
            moving_type_list=phase_types,
            moving_kwargs_list=[{} for _ in range(len(phase_types))],
            )
    
    @staticmethod
    def get_lr_wd_scale_schedulers(cfg, cfg_for_optimizer, train_loader) -> Tuple[ArbitraryScheduler, ArbitraryScheduler]:
        iters_per_epoch = len(train_loader)
        total_iters = cfg.trainer.epochs * iters_per_epoch
        lr_scale_scheduler = SchedulerUtils.get_arbitrary_scheduler(
            phase_epochs=cfg_for_optimizer.lr_scheduler.lr_phase_epochs,
            phase_steps=cfg_for_optimizer.lr_scheduler.lr_phase_steps,
            phase_scales=cfg_for_optimizer.lr_scheduler.lr_phase_scales,
            phase_types=cfg_for_optimizer.lr_scheduler.lr_phase_types,
            iters_per_epoch=iters_per_epoch,
            total_iters=total_iters,
            )
        wd_scale_scheduler = SchedulerUtils.get_arbitrary_scheduler(
            phase_epochs=cfg_for_optimizer.wd_scheduler.wd_phase_epochs,
            phase_steps=cfg_for_optimizer.wd_scheduler.wd_phase_steps,
            phase_scales=cfg_for_optimizer.wd_scheduler.wd_phase_scales,
            phase_types=cfg_for_optimizer.wd_scheduler.wd_phase_types,
            iters_per_epoch=iters_per_epoch,
            total_iters=total_iters,
            )
        return lr_scale_scheduler, wd_scale_scheduler
