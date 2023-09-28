import torch

from .simple_model import SimpleModel


class ModelManager(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.env.device)

    def build_model(self):
        if self.cfg.model.architecture == 'simple':
            model = SimpleModel(self.cfg).to(self.device)
        else:
            raise NotImplementedError(f'model architecture {self.cfg.model.architecture} not implemented.')
        print('model built successfully.')

        if hasattr(self.cfg.info, 'wandb_run'):
            if self.cfg.info.wandb_watch_model:
                self.cfg.info.wandb_run.watch(model, log='all', log_freq=self.cfg.info.wandb_watch_freq, log_graph=True)

        return model
