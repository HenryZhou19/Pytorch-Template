import torch

from .simple_loss import SimpleLoss
from .simple_metric import SimpleMetric


class CriterionManager(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.env.device)
 
    def build_criterion(self):
        loss_criterion = SimpleLoss(self.cfg).to(self.device)
        metric_criterion = SimpleMetric(self.cfg)
        print('criterion built successfully.')
        return loss_criterion, metric_criterion