import torch

from .simple_criterion import SimpleCriterion


class CriterionManager(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.env.device)
 
    def build_criterion(self):
        criterion = SimpleCriterion(self.cfg).to(self.device)
        print('criterion built successfully.')
        return criterion