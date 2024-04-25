from src.utils.misc import ModelMisc
from src.utils.optimizer.modules.warmup_scheduler import \
    WarmupCosineAnnealingMultiCycleLR

from .modules.gear_base import (TesterBase, TrainerBase, tester_register,
                                trainer_register)


@trainer_register('multi_cycle')
class Trainer(TrainerBase):
    def _train_mode(self):
        super()._train_mode()
        
    def _eval_mode(self):
        super()._eval_mode()
    
    def before_all_epochs(self, **kwargs):
        def clean_cycle_modules(modules):
            # remove the modules that are frozen
            clean_modules = []
            for module in modules:
                if module not in self.freeze_modules:
                    clean_modules.append(module)
            return clean_modules
        
        assert isinstance(self.lr_scheduler, WarmupCosineAnnealingMultiCycleLR)
        
        super().before_all_epochs(**kwargs)
        
        self.cycle_type = self.lr_scheduler.cycle_type
        self.cycle_modules_list = [clean_cycle_modules(cycle_modules) for cycle_modules in self.cfg.trainer.cycle_modules_list]
        
    def before_one_epoch(self, **kwargs):
        # self.epoch == 0 here before the first epoch
        super().before_one_epoch(**kwargs)
        # self.epoch == self.epoch + 1 after the "super.()..."
        
        if self.cycle_type != self.lr_scheduler.cycle_type:
            self.cycle_type = self.lr_scheduler.cycle_type
            if self.cfg.trainer.copy_ema_after_each_cycle:
                assert self.ema_model is not None, "EMA model is not initialized"
                self.ema_model.copy_params_from_ema_to_model()
            self.change_train_mode(self.model_without_ddp, self.cycle_type, self.cycle_modules_list)
        if self.epoch == 1:
            self.change_train_mode(self.model_without_ddp, self.cycle_type, self.cycle_modules_list)
        
    def after_one_epoch(self, **kwargs):
        super().after_one_epoch(**kwargs)
        
    def before_validation(self, **kwargs):
        super().before_validation(**kwargs)

    def after_validation(self, **kwargs):
        super().after_validation(**kwargs)
        
    def after_all_epochs(self, **kwargs):
        super().after_all_epochs(**kwargs)
    
    @staticmethod
    def change_train_mode(model_without_ddp, cycle_type, cycle_modules_list):
        for cycle_modules_idx in range(len(cycle_modules_list)):
            print('\n\n\n', cycle_modules_list[cycle_type])
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
                    verbose=True,
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
                    verbose=True,
                )


@tester_register('default')
class Tester(TesterBase):
    def _eval_mode(self):
        super()._eval_mode()
    
    def before_inference(self, **kwargs):
        super().before_inference(**kwargs)

    def after_inference(self, **kwargs):
        super().after_inference(**kwargs)
        