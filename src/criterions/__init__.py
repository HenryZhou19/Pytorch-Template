import torch

from src.utils.misc import ImportMisc

from .modules.criterion_base import CriterionBase, criterion_register

ImportMisc.import_current_dir_all(__file__, __name__)

class CriterionManager(object):
    def __init__(self, cfg, loggers) -> None:
        self.cfg = cfg
        self.loggers = loggers
        self.device = torch.device(cfg.env.device)
 
    def build_criterion(self) -> CriterionBase:
        criterion: CriterionBase = criterion_register.get(self.cfg.architecture)(self.cfg).to(self.device)
        criterion.untrainable_check()
        print('criterion built successfully.')
        return criterion