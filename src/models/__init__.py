import torch

from src.utils.simple_loss import SimpleLoss
from src.utils.simple_metric import SimpleMetric

from .simple_model import SimpleModel


class ModelManager(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.env.device)

    def build_model(self):
        model = SimpleModel(self.cfg).to(self.device)
        print('model built successfully.')

        if hasattr(self.cfg.info, 'wandb_run'):
            if self.cfg.info.wandb_watch_model:
                self.cfg.info.wandb_run.watch(model, log='all', log_freq=self.cfg.info.wandb_watch_freq, log_graph=True)

        return model
 
    def build_criterion(self):
        loss_criterion = SimpleLoss(self.cfg).to(self.device)
        metric_criterion = SimpleMetric(self.cfg)
        print('criterion built successfully.')
        return loss_criterion, metric_criterion