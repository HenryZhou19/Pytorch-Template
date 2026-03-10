from src.utils.register import get_registered_class, scan_register_classes

from .modules.tester_base import TesterBase
from .modules.trainer_base import TrainerBase

registered_trainers = scan_register_classes(py_dir=__path__[0], register_type='trainer_register')
registered_testers = scan_register_classes(py_dir=__path__[0], register_type='tester_register')

# from src.utils.misc import ImportMisc

# from .modules.tester_base import TesterBase, tester_register
# from .modules.trainer_base import TrainerBase, trainer_register

# ImportMisc.import_current_dir_all(__file__, __name__)


class GearManager:
    def __init__(self, cfg, loggers) -> None:
        self.cfg = cfg
        self.loggers = loggers
        
    def build_trainer(self, *args, **kwargs):
        # trainer: TrainerBase = trainer_register.get(self.cfg.trainer.trainer_choice)(self.cfg, self.loggers, *args, **kwargs)
        trainer: TrainerBase = get_registered_class(registered_trainers, self.cfg.trainer.trainer_choice, package='src.gears')(self.cfg, self.loggers, *args, **kwargs)
        return trainer
        
    def build_tester(self, *args, **kwargs):
        tester: TesterBase = get_registered_class(registered_testers, self.cfg.tester.tester_choice, package='src.gears')(self.cfg, self.loggers, *args, **kwargs)
        return tester
    
    def build_tester_model_only_mode(self, *args, **kwargs):
        tester: TesterBase = get_registered_class(registered_testers, self.cfg.tester.tester_choice, package='src.gears')(self.cfg, self.loggers, *args, **kwargs, model_only_mode=True)
        return tester
        