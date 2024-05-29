import torch

from src.utils.misc import *
from src.utils.optimizer.modules.warmup_scheduler import \
    WarmupCosineAnnealingMultiCycleLR

from .modules.trainer_base import TrainerBase, trainer_register


@trainer_register('multi_cycle')
class MultiCycleTrainer(TrainerBase):
    def _before_all_epochs(self, **kwargs):
        def _clean_cycle_modules(modules):
            # remove the modules that are frozen
            clean_modules = []
            for module in modules:
                if module not in self.freeze_modules:
                    clean_modules.append(module)
            return clean_modules
        
        assert isinstance(self.lr_scheduler, WarmupCosineAnnealingMultiCycleLR)
        
        super()._before_all_epochs(**kwargs)
        
        self.cycle_type = self.lr_scheduler.cycle_type
        self.cycle_modules_list = [_clean_cycle_modules(cycle_modules) for cycle_modules in self.cfg.trainer.cycle_modules_list]
        print(LoggerMisc.block_wrapper(f'Cycle modules list: {self.cycle_modules_list}'))
        self.min_hold_memory_mb = self.cfg.trainer.min_hold_memory_mb
        
    def _before_one_epoch(self, **kwargs):
        # self.epoch == 0 here before the first epoch
        super()._before_one_epoch(**kwargs)
        # self.epoch == self.epoch + 1 after the "super.()..."
        
        self.new_cycle = (self.epoch == 1)
        
        if self.cycle_type != self.lr_scheduler.cycle_type:
            self.cycle_type = self.lr_scheduler.cycle_type
            if self.cfg.trainer.copy_ema_after_each_cycle:
                assert self.ema_model is not None, "EMA model is not initialized"
                self.ema_model.copy_params_from_ema_to_model()
            
            self.new_cycle = True

        self._set_cycle_train_mode(self.model_without_ddp, self.cycle_type, self.cycle_modules_list)

        if self.new_cycle and DistMisc.is_dist_avail_and_initialized():
            if hasattr(self, 'memory_tensor'):
                del self.memory_tensor
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
    def _after_first_train_iter(self, **kwargs):
        super()._after_first_train_iter(**kwargs)
        
        if self.new_cycle and DistMisc.is_dist_avail_and_initialized():
            _, max_allocated_mb, reserved_mb, _ = TensorMisc.get_gpu_memory_usage(verbose=False)
            print(LoggerMisc.block_wrapper(f'Epoch {self.epoch}: Cycle Type {self.cycle_type}\n\tMax allocated memory: {max_allocated_mb:.2f} MB\n\tReserved memory: {reserved_mb:.2f} MB\n'))
            if reserved_mb < self.min_hold_memory_mb:
                self.memory_tensor = TensorMisc.allocate_memory_to_tensor(self.min_hold_memory_mb - reserved_mb)
    
    @staticmethod
    def _set_cycle_train_mode(model_without_ddp, cycle_type, cycle_modules_list):
        for cycle_modules_idx in range(len(cycle_modules_list)):
            if cycle_modules_idx == cycle_type:
                ModelMisc.train_or_eval_submodules(
                    model_without_ddp,
                    cycle_modules_list[cycle_modules_idx],
                    True,
                    verbose=False,
                )
                ModelMisc.unfreeze_or_freeze_submodules(
                    model_without_ddp,
                    cycle_modules_list[cycle_modules_idx],
                    True,
                    verbose=False,
                )
            else:
                ModelMisc.train_or_eval_submodules(
                    model_without_ddp,
                    cycle_modules_list[cycle_modules_idx],
                    False,
                    verbose=False,
                )
                ModelMisc.unfreeze_or_freeze_submodules(
                    model_without_ddp,
                    cycle_modules_list[cycle_modules_idx],
                    False,
                    verbose=False,
                )
