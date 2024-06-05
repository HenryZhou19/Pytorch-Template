import math
import time
from argparse import Namespace
from copy import deepcopy

import torch

from src.criterions import CriterionBase
from src.datasets.modules.data_module_base import DataLoaderX
from src.models import ModelBase
from src.utils.misc import *
from src.utils.progress_logger import *
from src.utils.register import Register

tester_register = Register('tester')

class TesterBase:
    def __init__(
        self,
        cfg: Namespace,
        loggers: Namespace,
        model_without_ddp: ModelBase,
        ema_model: torch.nn.Module,
        device: torch.device,
        criterion: CriterionBase=None,
        test_loader: DataLoaderX=None,
        model_only_mode=False,
        ) -> None:
        super().__init__()
        self.cfg = cfg
        self.loggers = loggers
        self.model = model_without_ddp
        self.ema_model = ema_model  # still in train mode (in ModelManager)
        self.device = device
        
        if self.ema_model is not None:
            self.ema_model.eval()
        
        if model_only_mode:
            self.nn_module_list = [self.model]
            
            if self.ema_model is not None:
                self.ema_model.eval()
        else:
            self.criterion = criterion
            self.test_loader = test_loader
            self.metrics = {}
            self.test_pbar = None
            self.test_len = len(self.test_loader)
            
            self.nn_module_list = [self.model, self.criterion]
            
            if self.ema_model is not None:
                self.ema_model.eval()
                print(LoggerMisc.block_wrapper('Using EMA model to infer. Setting EMA model and criterion to eval mode...', '='))
                self.ema_criterion = deepcopy(self.criterion)
                self.ema_criterion.eval()
                
            self.breath_time = self.cfg.tester.tester_breath_time  # XXX: avoid cpu being too busy
    
    def _get_pbar(self):
        # called in "before_inference"
        if DistMisc.is_main_process():
            test_pbar = LoggerMisc.MultiTQDM(
                total=self.test_len,
                dynamic_ncols=True,
                colour='green',
                position=0,
                maxinterval=math.inf,
            )
            test_pbar.set_description_str('Test ')
            print('')
            self.test_pbar = test_pbar
    
    def _load_model(self):
        # called in "before_inference"
        checkpoint = torch.load(self.cfg.tester.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        if self.ema_model is not None:
            assert checkpoint['ema_model'] is not None, 'ema_model is None in the checkpoint.'
            self.ema_model.load_state_dict(checkpoint['ema_model'])
        # print(f'{config.mode} mode: Loading pth from', path)
        print(LoggerMisc.block_wrapper(f'Loading pth from {self.cfg.tester.checkpoint_path}\nbest_trainer_metrics {checkpoint.get("best_metrics", {})}\nlast_trainer_metrics {checkpoint.get("last_metrics", {})}', '>'))
        if DistMisc.is_main_process():
            if 'epoch' in checkpoint.keys():
                print('Epoch:', checkpoint['epoch'])
                if hasattr(self.loggers, 'wandb_run'):
                    self.loggers.wandb_run.tags = self.loggers.wandb_run.tags + (f'Epoch: {checkpoint["epoch"]}',)
    
    def _eval_mode(self):
        # called in "before_inference"
        for nn_module in self.nn_module_list:
            nn_module.eval()
        # self.training = False
    
    def _before_inference(self, **kwargs):
        self._load_model()
        self._get_pbar()
        
        self._eval_mode()
    
    def _after_first_inference_iter(self, **kwargs):
        pass
    
    def _after_inference(self, **kwargs):
        LoggerMisc.logging(self.loggers,  'infer', self.metrics, None)
        
        if DistMisc.is_main_process():          
            self.test_pbar.close()
    
    def _forward(self, batch: dict):
        time.sleep(self.breath_time)
        
        batch: dict = TensorMisc.to(batch, self.device, non_blocking=self.cfg.env.pin_memory)
        inputs: dict = batch['inputs']
        targets: dict = batch['targets']
        
        with torch.no_grad():
            outputs = self.model(inputs)
            loss, metrics_dict = self.criterion(outputs, targets, infer_mode=True)
            
            if self.ema_model is not None:
                ema_outputs = self.ema_model(inputs)
                ema_loss, ema_metrics_dict = self.ema_criterion(ema_outputs, targets)
                # metrics_dict['ema_loss'] = ema_loss  # no need to show 'loss' & 'ema_loss' in inference
                for key, value in ema_metrics_dict.items():
                    metrics_dict[f'ema_{key}'] = value
            
        return outputs, loss, metrics_dict
    
    def _test(self):
        cfg = self.cfg
        
        mlogger = MetricLogger(
            cfg=cfg,
            loggers=self.loggers,
            pbar=self.test_pbar,  
            header='Test',
            )
        first_iter = True
        for batch in mlogger.log_every(self.test_loader):
            
            outputs, _, metrics_dict = self._forward(batch)
                
            mlogger.update_metrics(
                sample_count=batch['batch_size'],
                **metrics_dict,
                )
            
            if first_iter:
                first_iter = False
                self._after_first_inference_iter()
        
        mlogger.add_epoch_metrics(**self.criterion.get_epoch_metrics_and_reset())
        if hasattr(self, 'ema_criterion'):
            ema_epoch_metrics = {}
            raw_epoch_metrics = self.ema_criterion.get_epoch_metrics_and_reset()
            for k, v in raw_epoch_metrics.items():
                ema_epoch_metrics[f'ema_{k}'] = v
            mlogger.add_epoch_metrics(**ema_epoch_metrics)
        self.metrics = mlogger.output_dict(sync=True, final_print=True)
    
    def run(self):
        # prepare for 1. loading model; 2. progress bar
        self._before_inference()
        
        self._test()
        
        self._after_inference()
        
    def get_best_model_for_practical_use(self, verbose=True):
        self._load_model()
        self._eval_mode()
        if self.cfg.model.ema.ema_enabled and self.cfg.model.ema.ema_primary_criterion:
            if verbose:
                print(LoggerMisc.block_wrapper('using EMA model according to training config...'))
            return self.ema_model.ema_model
        else:
            if verbose:
                print(LoggerMisc.block_wrapper('NOT using EMA model according to training config...'))
            return self.model
    