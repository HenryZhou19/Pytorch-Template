from copy import deepcopy

import ema_pytorch
import torch
from ema_pytorch import EMA

from src.utils.misc import ImportMisc

from .modules.model_base import ModelBase, model_register

ImportMisc.import_current_dir_all(__file__, __name__)

class ModelManager(object):
    def __init__(self, cfg, loggers) -> None:
        self.cfg = cfg
        self.loggers = loggers
        self.device = torch.device(cfg.env.device)

    def build_model(self, verbose=True) -> ModelBase: 
        model: ModelBase = model_register.get(self.cfg.model.model_choice)(self.cfg).to(self.device)
        
        if verbose:
            print('model built successfully.')

            if hasattr(self.loggers, 'wandb_run'):
                if self.cfg.info.wandb.wandb_watch_model:
                    self.loggers.wandb_run.watch(model, log='all', log_freq=self.cfg.info.wandb.wandb_watch_freq, log_graph=True)

        return model
    
    def build_ema(self, model_without_ddp, verbose=True) -> ModelBase:
        if self.cfg.model.ema.ema_enabled:
            assert self.cfg.model.ema.ema_type == 'EMA', 'only support vanilla EMA for now.'
            
            ema_init_kwargs = {
                'model': model_without_ddp,
                'beta': self.cfg.model.ema.ema_beta,
                'update_after_step': self.cfg.model.ema.ema_update_after_step,
                'update_every': self.cfg.model.ema.ema_update_every,
                'include_online_model': False,
                }
            try:
                ema_model = deepcopy(model_without_ddp)
            except Exception as e:
                print(f'Warning: {e}')
                print('Build an EMA model from scratch and load state_dict instead.')
                ema_model = self.build_model(verbose=False)
                ema_model.load_state_dict(model_without_ddp.state_dict())
                
            ema_init_kwargs['ema_model'] = ema_model
            ema_container: EMA = ema_pytorch.__dict__.get(self.cfg.model.ema.ema_type)(**ema_init_kwargs).to(self.device)
            
            assert hasattr(ema_container.ema_model, 'ema_mode'), 'ema_container.ema_model doesn\'t have ema_mode attribute, which means the model is not a ModelBase instance.'
            ema_container.ema_model.set_ema_mode(True)
            if verbose:
                print('EMA_model built successfully.')
                
        else:
            ema_container = None
            if verbose:
                print('Not using EMA.')

        return ema_container

    def build_postprocessor(self, verbose=True):
        raise NotImplementedError