from src.utils.misc import ImportMisc

from .modules.gear_base import (TesterBase, TrainerBase, tester_register,
                                trainer_register)

ImportMisc.import_current_dir_all(__file__, __name__)

class GearManager(object):
    def __init__(self, cfg, loggers) -> None:
        self.cfg = cfg
        self.loggers = loggers
        
    def build_trainer(self, *args, **kwargs):
        trainer: TrainerBase = trainer_register.get(self.cfg.trainer.trainer_choice)(self.cfg, self.loggers, *args, **kwargs)
        return trainer
        
    def build_tester(self, *args, **kwargs):
        tester: TesterBase = tester_register.get(self.cfg.tester.tester_choice)(self.cfg, self.loggers, *args, **kwargs)
        return tester
        