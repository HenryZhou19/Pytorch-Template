from .modules.gear_base import (TesterBase, TrainerBase, tester_register,
                                trainer_register)


@trainer_register('default')
class Trainer(TrainerBase):
    def _train_mode(self):
        super()._train_mode()
        
    def _eval_mode(self):
        super()._eval_mode()
    
    def before_all_epochs(self, **kwargs):
        super().before_all_epochs(**kwargs)
    
    def before_one_epoch(self, **kwargs):
        super().before_one_epoch(**kwargs)
        
    def after_training_before_validation(self, **kwargs):
        super().after_training_before_validation(**kwargs)

    def after_validation(self, **kwargs):
        super().after_validation(**kwargs)
        
    def after_all_epochs(self, **kwargs):
        super().after_all_epochs(**kwargs)


@tester_register('default')
class Tester(TesterBase):
    def _eval_mode(self):
        super()._eval_mode()
    
    def before_inference(self, **kwargs):
        super().before_inference(**kwargs)

    def after_inference(self, **kwargs):
        super().after_inference(**kwargs)
        