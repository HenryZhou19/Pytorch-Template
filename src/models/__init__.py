import torch

from src.utils.misc import ImportMisc

from .modules.model_base import ModelBase, model_register

ImportMisc.import_current_dir_all(__file__, __name__)

class ModelManager(object):
    def __init__(self, cfg, loggers) -> None:
        self.cfg = cfg
        self.loggers = loggers
        self.device = torch.device(cfg.env.device)

    def build_model(self, verbose=True) -> ModelBase: 
        model: ModelBase = model_register.get(self.cfg.architecture)(self.cfg).to(self.device)
        
        if verbose:
            print('model built successfully.')

            if hasattr(self.loggers, 'wandb_run'):
                if self.cfg.info.wandb.wandb_watch_model:
                    self.loggers.wandb_run.watch(model, log='all', log_freq=self.cfg.info.wandb.wandb_watch_freq, log_graph=True)

        return model
