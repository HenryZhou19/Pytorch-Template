import math
import time
from argparse import Namespace
from copy import deepcopy
from functools import partial

import torch
from ema_pytorch.ema_pytorch import EMA

from src.criterions import CriterionBase
from src.datasets.modules.data_module_base import DataLoaderX
from src.models import ModelBase
from src.utils.misc import *
from src.utils.progress_logger import *
from src.utils.register import Register

tester_register = Register('tester')

class TesterBase:
    registered_name: str
    
    def __init__(
        self,
        cfg: Namespace,
        loggers: Namespace,
        model_without_ddp: ModelBase,
        ema_container: EMA,
        device: torch.device,
        criterion: CriterionBase=None,
        test_loader: DataLoaderX=None,
        model_only_mode=False,
        ) -> None:
        super().__init__()
        self.cfg = cfg
        self.loggers = loggers
        self.model = model_without_ddp
        self.ema_container = ema_container  # still in train mode (inited in ModelManager)
        self.device = device
        
        self.model.set_infer_mode(True)
        
        if model_only_mode:
            self.nn_module_list = [self.model]
            
            if self.ema_container is not None:
                self.ema_container.ema_model.set_infer_mode(True)
                self.ema_container.eval()
        else:
            self.criterion = criterion
            assert hasattr(self.criterion, 'infer_mode'), 'criterion doesn\'t have infer_mode attribute, which means the criterion is not a CriterionBase instance.'
            self.criterion.set_infer_mode(True)
            self.test_loader = test_loader
            self.test_metrics = {}
            self.test_pbar = None
            self.test_len = len(self.test_loader)
            
            self.nn_module_list = [self.model, self.criterion]
            
            if self.ema_container is not None:
                self.ema_container.ema_model.set_infer_mode(True)
                self.ema_container.eval()
                print(LoggerMisc.block_wrapper('Using EMA model to infer. Setting EMA model and criterion to eval mode...', '='))
                self.ema_criterion = deepcopy(self.criterion)
                assert hasattr(self.ema_criterion, 'ema_mode'), 'ema_criterion doesn\'t have ema_mode attribute, which means the criterion is not a CriterionBase instance.'
                self.ema_criterion.set_ema_mode(True)
                assert hasattr(self.ema_criterion, 'infer_mode'), 'ema_criterion doesn\'t have infer_mode attribute, which means the criterion is not a CriterionBase instance.'
                self.ema_criterion.set_infer_mode(True)
                self.ema_criterion.eval()
                
            self.breath_time = self.cfg.tester.tester_breath_time  # XXX: avoid cpu being too busy
            self.ema_only = self.cfg.tester.ema_only
            self._init_autocast()
            
    def _init_autocast(self):
        if self.cfg.env.amp.amp_mode == 'fp16':
            dtype = torch.float16
        elif self.cfg.env.amp.amp_mode == 'bf16':
            dtype = torch.bfloat16
        else:
            raise ValueError(f'Unknown amp.amp_mode: {self.cfg.env.amp.amp_mode}')
        inference_amp_enabled = self.cfg.env.amp.amp_enabled and self.cfg.env.amp.amp_inference
        self.inference_autocast = partial(torch.cuda.amp.autocast, enabled=inference_amp_enabled, dtype=dtype)
    
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
        if self.ema_container is not None:
            assert 'ema_container' in checkpoint or 'ema_model' in checkpoint, 'checkpoint does not contain "ema_container" or "ema_model".'
            if 'ema_container' in checkpoint:
                self.ema_container.load_state_dict(checkpoint['ema_container'])
            else:  # FIXME: deprecated
                self.ema_container.load_state_dict(checkpoint['ema_model'])
        # print(f'{config.mode} mode: Loading pth from', path)
        print(LoggerMisc.block_wrapper(f'Loading pth from {self.cfg.tester.checkpoint_path}\nbest_val_metrics {checkpoint.get("best_val_metrics", {})}\nlast_val_metrics {checkpoint.get("last_val_metrics", {})}', '>'))
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
        LoggerMisc.logging(self.loggers,  'infer', self.test_metrics, None)
        
        if DistMisc.is_main_process():          
            self.test_pbar.close()
    
    def _forward(self, batch: dict):
        time.sleep(self.breath_time)
        
        batch: dict = TensorMisc.to(batch, self.device, non_blocking=self.cfg.env.pin_memory)
        inputs: dict = batch['inputs']
        targets: dict = batch['targets']
        
        with torch.no_grad():
            with self.inference_autocast():
                if self.ema_only:
                    assert self.ema_container is not None, 'ema_container is None when ema_only is True.'
                    outputs = {}
                    loss = torch.nan
                    metrics_dict = {}
                else:
                    outputs = self.model(inputs)
                    loss, metrics_dict = self.criterion(outputs, targets)
                
                if self.ema_container is not None:
                    ema_outputs = self.ema_container(inputs)
                    _, ema_metrics_dict = self.ema_criterion(ema_outputs, targets)
                    outputs.update(LoggerMisc.set_dict_key_prefix(ema_outputs, 'ema_'))
                    metrics_dict.update(ema_metrics_dict)
            
        return outputs, loss, metrics_dict
    
    def _test(self):
        cfg = self.cfg
        
        mlogger = MetricLogger(
            cfg=cfg,
            loggers=self.loggers,
            pbar=self.test_pbar,  
            header='Test',
            )
        mlogger.add_metrics([{'loss': ValueMetric(high_prior=True)}])
        first_iter = True
        for batch in mlogger.log_every(self.test_loader):
            
            _, loss, metrics_dict = self._forward(batch)
                
            mlogger.update_metrics(
                sample_count=batch['batch_size'],
                loss=loss,
                **metrics_dict,
                )
            
            if first_iter:
                first_iter = False
                self._after_first_inference_iter()
        
        mlogger.add_epoch_metrics(**self.criterion.forward_epoch_metrics())
        if hasattr(self, 'ema_criterion'):
            mlogger.add_epoch_metrics(**self.ema_criterion.forward_epoch_metrics())
        self.test_metrics = mlogger.output_dict(sync=True, final_print=True)
        
    def _print_module_states(self, prefix):
        print(f'\n[{prefix}]')
        DistMisc.avoid_print_mess()
        print(f'\tRank {DistMisc.get_rank()}:', force=True)
        print(f'\t\tOnline:', force=True)
        self.model.print_states(prefix='\t\t\t')
        self.criterion.print_states(prefix='\t\t\t')
        if self.ema_container is not None:
            print(f'\t\tEMA:', force=True)
            self.ema_container.ema_model.print_states(prefix='\t\t\t')
            self.ema_criterion.print_states(prefix='\t\t\t')
    
    def run(self):
        # prepare for 1. loading model; 2. progress bar
        self._before_inference()
        
        if self.cfg.info.print_module_states:
            self._print_module_states('Test')
        self._test()
        
        self._after_inference()
        
    def get_model_for_practical_use(self, get_ema_model=False, verbose=True):
        self._load_model()
        self._eval_mode()
        if get_ema_model:
            assert self.ema_container is not None, 'ema_container is None when get_ema_model is True.'
            string_to_print = 'Using the EMA model...'
            return_model = self.ema_container.ema_model
        else:
            string_to_print = 'Using the online model...'
            return_model = self.model
        if self.cfg.model.ema.ema_enabled and self.cfg.model.ema.ema_primary_criterion:
            string_to_print += '\nIn trainer, ema_primary_criterion is True, so using the "best" checkpoint and "get_ema_model=True" is recommended.'
        else:
            string_to_print += '\nIn trainer, ema_primary_criterion is False, so using the "best" checkpoint and "get_ema_model=False" is recommended.'
        
        if verbose:
            print(LoggerMisc.block_wrapper(string_to_print, '='))
        return return_model
    