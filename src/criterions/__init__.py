import torch

from src.utils.misc import ImportMisc

from .modules.criterion_base import register

ImportMisc.import_current_dir_all(__file__, __name__)

class CriterionManager(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.env.device)
 
    def build_criterion(self):
        criterion = register.get(self.cfg.model.architecture)(self.cfg).to(self.device)
        print('criterion built successfully.')
        return criterion