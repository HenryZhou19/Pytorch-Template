import math
import os
import time
from argparse import Namespace
from copy import deepcopy
from glob import glob

import torch
from ema_pytorch import EMA

from src.criterions import CriterionBase
from src.datasets.modules.data_module_base import DataLoaderX
from src.models import ModelBase
from src.utils.misc import *
from src.utils.progress_logger import *
from src.utils.register import Register

trainer_register = Register('trainer')

class TrainerBase:
    def __init__(
        self,
        cfg: Namespace,
        loggers: Namespace,
        model: ModelBase,
        ema_model: EMA,
        criterion: CriterionBase,
        train_loader: DataLoaderX,
        val_loader: DataLoaderX,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        scaler: torch.cuda.amp.GradScaler,
        device: torch.device,
        ) -> None:
        super().__init__()
        self.cfg = cfg
        self.loggers = loggers
        self.model = model
        self.model_without_ddp = model.module if cfg.env.distributed else model
        self.ema_model = ema_model  # still in train mode (in ModelManager)
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        self.device = device
        self.start_epoch = 1
        self.epoch = self.start_epoch - 1
        self.train_outputs = {}
        self.metrics = {}
        self.best_metrics = {}
        self.train_pbar = None
        self.val_pbar = None
        self.val_epoch_list = self._get_val_epochs()
        self.dist_eval = self.cfg.trainer.dist_eval
        
        assert self.cfg.trainer.grad_accumulation > 0 and isinstance(self.cfg.trainer.grad_accumulation, int), 'grad_accumulation should be a positive integer.'
        
        self.gradient_accumulation_steps = self.cfg.trainer.grad_accumulation
        self.do_gradient_accumulation = self.gradient_accumulation_steps > 1
        self.max_grad_norm = self.cfg.trainer.max_grad_norm
        self.train_len = len(self.train_loader)
        self.val_len = len(self.val_loader)
        
        self.trained_iters = 0
        self.total_iters = self.cfg.trainer.epochs * self.train_len
          
        self.nn_module_list = [self.model, self.criterion]
        self.freeze_modules = getattr(self.cfg.trainer, 'freeze_modules', [])
        self.training = True
        
        if self.ema_model is not None:
            print(LoggerMisc.block_wrapper('Using EMA model to evaluate. Setting EMA model and criterion to eval mode...', '='))
            self.ema_model.eval()
            self.ema_criterion = deepcopy(self.criterion)
            self.ema_criterion.eval()
        
        self.breath_time = self.cfg.trainer.trainer_breath_time  # XXX: avoid cpu being too busy
        self.checkpoint_save_interval = self.cfg.trainer.checkpoint_save_interval
        self.checkpoint_reserve_interval = self.cfg.trainer.checkpoint_save_interval * self.cfg.trainer.checkpoint_reserve_factor
    
    @property
    def epoch_loop(self):
        return range(self.start_epoch, self.cfg.trainer.epochs + 1)
    
    @property
    def lr_groups(self):
        return {'lr_' + param_group['group_name']: param_group['lr']
                for param_group in self.optimizer.param_groups if not param_group['group_name'].endswith('_no_wd')}
    
    @property
    def wd_groups(self):
        return {'wd_' + param_group['group_name']: param_group['weight_decay']
                for param_group in self.optimizer.param_groups if not param_group['group_name'].endswith('_no_wd')}
    
    def _get_val_epochs(self):
        if self.cfg.trainer.eval_freq <= 0:
            val_epochs = [self.cfg.trainer.epochs]
        else:
            val_epochs = list(range(self.cfg.trainer.eval_freq, self.cfg.trainer.epochs + 1, self.cfg.trainer.eval_freq))
            if val_epochs[-1] != self.cfg.trainer.epochs:
                val_epochs.append(self.cfg.trainer.epochs)
        return val_epochs
    
    def _get_pbar(self):
        # called in "before_all_epochs"
        if DistMisc.is_main_process():
            epoch_finished = self.start_epoch - 1
            train_pbar = LoggerMisc.MultiTQDM(
                total=self.total_iters if self.cfg.info.global_tqdm else self.train_len,
                dynamic_ncols=True,
                colour='green',
                position=0,
                maxinterval=math.inf,
                initial=epoch_finished * self.train_len,
            )
            train_pbar.set_description_str('Train')
            print('')
            val_pbar = LoggerMisc.MultiTQDM(
                total=len(self.val_epoch_list) * self.val_len if self.cfg.info.global_tqdm else self.val_len,
                dynamic_ncols=True,
                colour='green',
                position=0,
                maxinterval=math.inf,
                initial=len([x for x in self.val_epoch_list if x <= epoch_finished]) * self.val_len,
            )
            val_pbar.set_description_str('Eval ')
            print('')
            self.train_pbar = train_pbar
            self.val_pbar = val_pbar
    
    def _resume_training(self):
        # called in "before_all_epochs"
        if self.cfg.trainer.resume:
            checkpoint_path = glob(os.path.join(self.cfg.info.work_dir, 'checkpoint_last_epoch_*.pth'))
            print(LoggerMisc.block_wrapper(f'loading the checkpoint from {checkpoint_path}', '>'))
            assert len(checkpoint_path) == 1, f'Found {len(checkpoint_path)} checkpoints, please check.'
            checkpoint = torch.load(checkpoint_path[0], map_location='cpu')
            self.model_without_ddp.load_state_dict(checkpoint['model'])
            if self.ema_model is not None:
                assert checkpoint['ema_model'] is not None, 'ema_model is None in the checkpoint.'
                self.ema_model.load_state_dict(checkpoint['ema_model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if self.cfg.env.amp.amp_enabled:
                self.scaler.load_state_dict(checkpoint['scaler'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_metrics = checkpoint.get('best_metrics', {})
            self.metrics = checkpoint.get('last_metrics', {})
            self.trained_iters = checkpoint['epoch'] * self.train_len
            self.epoch = self.start_epoch - 1  # will be the same as {checkpoint['epoch'] + 1} by doing '+1' in "before_one_epoch"
        else:
            print(LoggerMisc.block_wrapper('New trainer.', '>'))
        print(f'Start from epoch: {self.start_epoch}')
        return self.cfg.trainer.resume
    
    def _load_pretrained_models(self):
        def _load_pretrained_model(model_path, pretrain_model_name):
            if self.ema_model is not None and self.cfg.trainer.load_from_ema:
                print(f'\nLoading {pretrain_model_name} (key="ema_model") from {model_path}')
                state_dict = torch.load(model_path, map_location='cpu')['ema_model']
                state_dict.pop('initted', None)
                state_dict.pop('step', None)
                ModelMisc.load_state_dict_with_more_info(
                    self.ema_model,
                    state_dict,
                    strict=False,
                    print_keys_level=2,
                    )
                self.ema_model.copy_params_from_ema_to_model()
            else:
                print(f'\nLoading {pretrain_model_name} (key="model") from {model_path}')
                ModelMisc.load_state_dict_with_more_info(
                    self.model_without_ddp,
                    torch.load(model_path, map_location='cpu')['model'],
                    strict=False,
                    print_keys_level=1,
                    )
                self.ema_model.copy_params_from_model_to_ema()
        
        if getattr(self.cfg.trainer, 'pretrained_models', None) is not None:
            for pretrain_model_name, pretrained_model_path in ConfigMisc.nested_namespace_to_nested_dict(self.cfg.trainer.pretrained_models).items():
                if pretrained_model_path is not None:
                    _load_pretrained_model(pretrained_model_path, pretrain_model_name)
    
    def _save_checkpoint(self):
        # called in "after_one_epoch"
        if DistMisc.is_main_process():
            epoch_finished = self.epoch
            
            if epoch_finished % self.checkpoint_save_interval == 0:
                save_files = {
                    'model': self.model_without_ddp.state_dict(),
                    'ema_model': self.ema_model.state_dict() if self.ema_model is not None else None,
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                    'last_metrics': self.metrics,
                    'epoch': epoch_finished,
                }
                if self.cfg.env.amp.amp_enabled:
                    save_files.update({
                        'scaler': self.scaler.state_dict()
                    })
                last = glob(os.path.join(self.cfg.info.work_dir, 'checkpoint_last_epoch_*.pth'))
                last_saved_epoch = max([int(os.path.basename(path).split('_')[-1].split('.')[0]) for path in last] + [0])
                if last_saved_epoch == 0 or (self.checkpoint_reserve_interval > 0 and last_saved_epoch % self.checkpoint_reserve_interval == 0):
                    torch.save(save_files, os.path.join(self.cfg.info.work_dir, f'checkpoint_last_epoch_{epoch_finished}.pth'))
                else:
                    last_path = os.path.join(self.cfg.info.work_dir, f'checkpoint_last_epoch_{last_saved_epoch}.pth')
                    torch.save(save_files, last_path)
                    os.rename(last_path, os.path.join(self.cfg.info.work_dir, f'checkpoint_last_epoch_{epoch_finished}.pth'))
    
    def _save_best_checkpoint(self):
        # called in "after_validation"
        if DistMisc.is_main_process():
            epoch_finished = self.epoch
            
            self.best_metrics, save_flag = self.criterion.choose_best(
                self.metrics, self.best_metrics
            )
            if save_flag:
                save_files = {
                    'model': self.model_without_ddp.state_dict(),
                    'ema_model': self.ema_model.state_dict() if self.ema_model is not None else None,
                    'best_metrics': self.best_metrics,
                    'epoch': epoch_finished,
                }
                
                best = glob(os.path.join(self.cfg.info.work_dir, 'checkpoint_best_epoch_*.pth'))
                assert len(best) <= 1
                if len(best) == 1:
                    torch.save(save_files, best[0])
                    os.rename(best[0], os.path.join(self.cfg.info.work_dir, f'checkpoint_best_epoch_{epoch_finished}.pth'))
                else:
                    torch.save(save_files, os.path.join(self.cfg.info.work_dir, f'checkpoint_best_epoch_{epoch_finished}.pth'))
    
    def _train_mode(self):
        # called in "before_one_epoch"
        for nn_module in self.nn_module_list:
            nn_module.train()
            
        ModelMisc.train_or_eval_submodules(
            self.model_without_ddp,
            self.freeze_modules,
            False,
        )
        self.training = True
    
    def _eval_mode(self):
        # called in "before_validation"
        for nn_module in self.nn_module_list:
            nn_module.eval()
        self.training = False
    
    def _before_all_epochs(self, **kwargs):
        is_resumed = self._resume_training()
        if not is_resumed:
            self._load_pretrained_models()
            
        ModelMisc.unfreeze_or_freeze_submodules(
            self.model_without_ddp,
            self.freeze_modules,
            False,
            )
            
        ModelMisc.show_model_info(self.cfg, self)
        self._get_pbar()
    
    def _before_one_epoch(self, **kwargs):
        self.epoch += 1
        if self.cfg.env.distributed:
            # shuffle data for each epoch (here needs epoch start from 0)
            self.train_loader.sampler_set_epoch(self.epoch - 1)  
        
        DistMisc.barrier()

        if DistMisc.is_main_process():
            if self.cfg.info.global_tqdm:
                self.train_pbar.unpause()
            else :
                self.train_pbar.reset()
                self.val_pbar.reset()
          
        self.step_count = 0
        self.optimizer.zero_grad()
        self._train_mode()
    
    def _after_first_train_iter(self, **kwargs):
        pass
    
    def _after_one_epoch(self, **kwargs):
        LoggerMisc.logging(self.loggers, 'train_epoch', self.train_outputs, self.trained_iters)
        
        self._save_checkpoint()
    
    def _before_validation(self, **kwargs):
        DistMisc.barrier()
        
        if DistMisc.is_main_process():
            self.val_pbar.unpause()
            
        self._eval_mode()
    
    def _after_first_validation_iter(self, **kwargs):
        pass
    
    def _after_validation(self, **kwargs):
        LoggerMisc.logging(self.loggers, 'val_epoch', self.metrics, self.trained_iters)
        
        self._save_best_checkpoint()
    
    def _after_all_epochs(self, **kwargs):
        DistMisc.barrier()
        
        if DistMisc.is_main_process():
            self.train_pbar.close()
            self.val_pbar.close() 
    
    def _forward(self, batch: dict):
        time.sleep(self.breath_time)
        
        batch: dict = TensorMisc.to(batch, self.device, non_blocking=self.cfg.env.pin_memory)
        inputs: dict = batch['inputs']
        targets: dict = batch['targets']
        
        inputs['train_progress'] = self.trained_iters / self.total_iters
        
        if self.training:
            with torch.cuda.amp.autocast(enabled=self.scaler is not None, dtype=getattr(self.scaler, 'custom_dtype', torch.float16)):
                outputs = self.model(inputs)
                loss, metrics_dict = self.criterion(outputs, targets)
        else:
            with torch.no_grad():
                outputs = self.model(inputs)
                loss, metrics_dict = self.criterion(outputs, targets)
                
                if self.ema_model is not None:
                    ema_outputs = self.ema_model(inputs)
                    ema_loss, ema_metrics_dict = self.ema_criterion(ema_outputs, targets)
                    metrics_dict['ema_loss'] = ema_loss
                    for key, value in ema_metrics_dict.items():
                        metrics_dict[f'ema_{key}'] = value
            
        return outputs, loss, metrics_dict
    
    def _backward_and_step(self, loss: torch.Tensor):
        def _backward():
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
        def _optimize():
            if self.scaler is not None:
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()
            self.optimizer.zero_grad()
            if self.ema_model is not None:
                self.ema_model.update()
        
        if not math.isfinite(loss) and self.scaler is None:
            LoggerMisc.get_wandb_pid(kill_all=True)
            raise ValueError(f'Rank {DistMisc.get_rank()}: Loss is {loss}, stopping training.')
        
        if self.do_gradient_accumulation:
            loss /= self.gradient_accumulation_steps  # Assume that all losses are mean-reduction. (Otherwise meaningless)
            self.step_count += 1
        
        _backward()
        
        if self.do_gradient_accumulation and (self.trained_iters % self.train_len != 0):
            if self.step_count % self.gradient_accumulation_steps == 0:
                _optimize()
                self.step_count = 0
        else:
            _optimize()
        
        self.lr_scheduler.step()  # update special lr_scheduler after each iter
        self.trained_iters += 1
        
    def _train_one_epoch(self):
        cfg = self.cfg
        loggers = self.loggers
        
        mlogger = MetricLogger(
            cfg=cfg,
            loggers=loggers,
            pbar=self.train_pbar,  
            header='Train',
            epoch_str=f'epoch: [{self.epoch}/{cfg.trainer.epochs}]',
            )
        mlogger.add_metrics([{
            'loss': ValueMetric(prior=True),
            **{lr_group: ValueMetric(format='{value:.2e}', final_format='[{min:.2e}, {max:.2e}]', no_sync=True) for lr_group in self.lr_groups.keys()},
            'epoch': ValueMetric(window_size=1, no_print=True, no_sync=True)
            }])
        first_iter = True
        for batch in mlogger.log_every(self.train_loader):
            
            _, loss, metrics_dict = self._forward(batch)
            
            mlogger.update_metrics(
                sample_count=batch['batch_size'],
                **self.lr_groups,
                loss=loss,
                **metrics_dict,
            )
            
            self._backward_and_step(loss)
            
            mlogger.update_metrics(
                epoch=self.trained_iters / self.train_len,
            )
            
            if cfg.info.iter_log_freq > 0:
                if self.trained_iters % (cfg.info.iter_log_freq * cfg.trainer.grad_accumulation) == 0:
                    LoggerMisc.logging(loggers, 'train_iter', mlogger.output_dict(no_avg_list=['all']), int(self.trained_iters / cfg.trainer.grad_accumulation))
            
            if first_iter:
                first_iter = False
                self._after_first_train_iter()
        
        mlogger.add_epoch_metrics(**self.criterion.get_epoch_metrics_and_reset())
        self.train_outputs = mlogger.output_dict(no_avg_list=[*self.lr_groups.keys(), 'epoch'], sync=True, final_print=True)
    
    def _evaluate(self):
        cfg = self.cfg
        
        mlogger = MetricLogger(
            cfg=cfg,
            loggers=self.loggers,
            pbar=self.val_pbar,  
            header='Eval ',
            epoch_str=f'epoch: [{self.epoch}/{cfg.trainer.epochs}]',
            )
        mlogger.add_metrics([{'loss': ValueMetric(prior=True)}])
        first_iter = True
        for batch in mlogger.log_every(self.val_loader):
            
            _, loss, metrics_dict = self._forward(batch)
            
            mlogger.update_metrics(
                sample_count=batch['batch_size'],
                loss=loss,
                **metrics_dict,
            )
            
            if first_iter:
                first_iter = False
                self._after_first_validation_iter()
        
        mlogger.add_epoch_metrics(**self.criterion.get_epoch_metrics_and_reset())
        if hasattr(self, 'ema_criterion'):
            ema_epoch_metrics = {}
            raw_epoch_metrics = self.ema_criterion.get_epoch_metrics_and_reset()
            for k, v in raw_epoch_metrics.items():
                ema_epoch_metrics[f'ema_{k}'] = v
            mlogger.add_epoch_metrics(**ema_epoch_metrics)
        self.metrics = mlogger.output_dict(sync=self.dist_eval, final_print=True)
    
    def run(self):
        # train and val
        # prepare for 1. resumed training if needed; 2. show model information; 3. progress bar;
        self._before_all_epochs()
        
        for _ in self.epoch_loop:
            self._before_one_epoch()
            
            self._train_one_epoch()
            
            self._after_one_epoch()
            
            if self.epoch in self.val_epoch_list:
                
                self._before_validation()
                
                if self.dist_eval or DistMisc.is_main_process():
                    self._evaluate()
                    
                self._after_validation()
                
                if self.cfg.special.debug == 'one_val_epoch':
                    PortalMisc.end_everything(self.cfg, self.loggers, force=True)
                
            if self.cfg.special.debug == 'one_epoch':
                PortalMisc.end_everything(self.cfg, self.loggers, force=True)
                
        self._after_all_epochs()
