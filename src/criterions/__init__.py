import torch

from src.utils.misc import ImportMisc

from .modules.criterion_base import CriterionBase, criterion_register

ImportMisc.import_current_dir_all(__file__, __name__)

class CriterionManager:
    def __init__(self, cfg, loggers) -> None:
        self.cfg = cfg
        self.loggers = loggers
        self.device = torch.device(cfg.env.device)
 
    def build_criterion(self) -> CriterionBase:
        criterion_choice = getattr(self.cfg.criterion, 'criterion_choice', 'default')
        criterion_choice = self.cfg.model.model_choice if criterion_choice == 'default' else criterion_choice
        criterion: CriterionBase = criterion_register.get(criterion_choice)(self.cfg).to(self.device)
        criterion.untrainable_check()
        print('criterion built successfully.')
        return criterion