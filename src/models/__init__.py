import torch

from src.utils.misc import ImportMisc

from .modules.model_base import register

ImportMisc.import_current_dir_all(__file__, __name__)

class ModelManager(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.env.device)

    def build_model(self): 
        model = register.get(self.cfg.model.architecture)(self.cfg).to(self.device)
        print('model built successfully.')

        if hasattr(self.cfg.info, 'wandb_run'):
            if self.cfg.info.wandb_watch_model:
                self.cfg.info.wandb_run.watch(model, log='all', log_freq=self.cfg.info.wandb_watch_freq, log_graph=True)

        return model
