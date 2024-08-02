import math
import os
import time
import warnings
from argparse import Namespace
from copy import deepcopy
from functools import partial
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
    registered_name: str
    
    def __init__(
        self,
        cfg: Namespace,
        loggers: Namespace,
        model: ModelBase,
        ema_container: EMA,
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
        self.ema_container = ema_container  # still in train mode (in ModelManager)
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
        self.last_val_metrics = {}
        self.best_val_metrics = {}
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
        
        if self.ema_container is not None:
            print(LoggerMisc.block_wrapper('Using EMA model to evaluate. Setting EMA model and criterion to eval mode...', '='))
            self.ema_container.eval()
            self.ema_criterion = deepcopy(self.criterion)
            assert hasattr(self.ema_criterion, 'ema_mode'), 'ema_criterion doesn\'t have ema_mode attribute, which means the criterion is not a CriterionBase instance.'
            self.ema_criterion.ema_mode = True
            self.ema_criterion.eval()
        
        self.breath_time = self.cfg.trainer.trainer_breath_time  # XXX: avoid cpu being too busy
        self.checkpoint_last_interval = self.cfg.trainer.checkpoint_last_interval  # save the last checkpoint every {checkpoint_last_interval} epochs (keep latest)
        assert self.checkpoint_last_interval > 0, 'checkpoint_last_interval should be a positive integer.'
        self.checkpoint_keep_interval = self.cfg.trainer.checkpoint_keep_interval  # save the checkpoint every {checkpoint_keep_interval} epochs (keep all)
        if self.checkpoint_keep_interval > 0:
            os.makedirs(os.path.join(self.cfg.info.work_dir, 'checkpoint_keep_storage'), exist_ok=True)
        self._init_autocast()
    
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
        
    def _init_autocast(self):
        if self.cfg.env.amp.amp_mode == 'fp16':
            dtype = torch.float16
        elif self.cfg.env.amp.amp_mode == 'bf16':
            dtype = torch.bfloat16
        else:
            raise ValueError(f'Unknown amp.amp_mode: {self.cfg.env.amp.amp_mode}')
        train_amp_enabled = self.cfg.env.amp.amp_enabled
        val_amp_enabled = self.cfg.env.amp.amp_enabled and self.cfg.env.amp.amp_val
        self.train_autocast = partial(torch.cuda.amp.autocast, enabled=train_amp_enabled, dtype=dtype)
        self.val_autocast = partial(torch.cuda.amp.autocast, enabled=val_amp_enabled, dtype=dtype)
    
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
            if self.ema_container is not None:
                assert 'ema_container' in checkpoint or 'ema_model' in checkpoint, 'checkpoint does not contain "ema_container" or "ema_model".'
                if 'ema_container' in checkpoint:
                    self.ema_container.load_state_dict(checkpoint['ema_container'])
                else:  # FIXME: deprecated
                    self.ema_container.load_state_dict(checkpoint['ema_model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if self.scaler is not None:
                assert 'scaler' in checkpoint, 'checkpoint does not contain "scaler".'
                self.scaler.load_state_dict(checkpoint['scaler'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_metrics = checkpoint.get('best_val_metrics', {})
            self.last_val_metrics = checkpoint.get('last_val_metrics', {})
            self.trained_iters = checkpoint['epoch'] * self.train_len
            self.epoch = self.start_epoch - 1  # will be the same as {checkpoint['epoch'] + 1} by doing '+1' in "before_one_epoch"
        else:
            print(LoggerMisc.block_wrapper('New trainer.', '>'))
        print(f'Start from epoch: {self.start_epoch}')
        return self.cfg.trainer.resume
    
    def _load_pretrained_models(self):
        def _load_pretrained_model(model_path, pretrain_model_name):
            if self.ema_container is not None and self.cfg.trainer.load_from_ema:
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'ema_container' in checkpoint:
                    key = 'ema_container'
                else:  # FIXME: deprecated
                    key = 'ema_model'
                print(f'\nLoading {pretrain_model_name} (key="{key}") from {model_path}')
                state_dict = checkpoint[key]
                state_dict.pop('initted', None)
                state_dict.pop('step', None)
                ModelMisc.load_state_dict_with_more_info(
                    self.ema_container,
                    state_dict,
                    strict=False,
                    print_keys_level=2,
                    )
                self.ema_container.copy_params_from_ema_to_model()
            else:
                print(f'\nLoading {pretrain_model_name} (key="model") from {model_path}')
                ModelMisc.load_state_dict_with_more_info(
                    self.model_without_ddp,
                    torch.load(model_path, map_location='cpu')['model'],
                    strict=False,
                    print_keys_level=1,
                    )
                self.ema_container.copy_params_from_model_to_ema()
        
        if getattr(self.cfg.trainer, 'pretrained_models', None) is not None:
            for pretrain_model_name, pretrained_model_path in ConfigMisc.nested_namespace_to_nested_dict(self.cfg.trainer.pretrained_models).items():
                if pretrained_model_path is not None:
                    _load_pretrained_model(pretrained_model_path, pretrain_model_name)
    
    @staticmethod
    def _save_or_update_checkpoint(save_dict, work_dir, epoch_finished, label):
        # label: 'last' or 'best'
        checkpoint_path_list = glob(os.path.join(work_dir, f'checkpoint_{label}_epoch_*.pth'))
        if len(checkpoint_path_list) > 1:
            warnings.warn(f'Found {len(checkpoint_path_list)} {label} checkpoints, please check.')
        max_saved_temp_epoch = max([int(os.path.basename(checkpoint_path).split('_')[-1].split('.')[0]) for checkpoint_path in checkpoint_path_list] + [0])
        new_path = os.path.join(work_dir, f'checkpoint_{label}_epoch_{epoch_finished}.pth')
        if max_saved_temp_epoch == 0:
            torch.save(save_dict, new_path)
        else:
            old_path = os.path.join(work_dir, f'checkpoint_{label}_epoch_{max_saved_temp_epoch}.pth')
            torch.save(save_dict, old_path)
            os.rename(old_path, new_path)
    
    def _save_checkpoint(self):
        # called in "after_one_epoch"
        if DistMisc.is_main_process():
            epoch_finished = self.epoch
            save_last = epoch_finished % self.checkpoint_last_interval == 0
            save_keep = epoch_finished % self.checkpoint_keep_interval == 0 if self.checkpoint_keep_interval > 0 else False
            
            if save_last or save_keep:
                save_dict = {
                    'model': self.model_without_ddp.state_dict(),
                    'ema_container': self.ema_container.state_dict() if self.ema_container is not None else None,
                    'scaler': self.scaler.state_dict() if self.scaler is not None else None,
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                    'last_val_metrics': self.last_val_metrics,
                    'epoch': epoch_finished,
                }
            if save_last:
                self._save_or_update_checkpoint(save_dict, self.cfg.info.work_dir, epoch_finished, 'last')
            if save_keep:
                keep_path = os.path.join(self.cfg.info.work_dir, f'checkpoint_keep_storage/checkpoint_keep_epoch_{epoch_finished}.pth')
                torch.save(save_dict, keep_path)
    
    def _save_best_only_model_checkpoint(self):
        # called in "after_validation"
        self.best_val_metrics, last_is_best = self.criterion.choose_best(
            self.last_val_metrics, self.best_val_metrics
            )
        
        if DistMisc.is_main_process():
            epoch_finished = self.epoch
            
            if last_is_best:
                save_dict = {
                    'model': self.model_without_ddp.state_dict(),
                    'ema_container': self.ema_container.state_dict() if self.ema_container is not None else None,
                    'best_val_metrics': self.best_val_metrics,
                    'epoch': epoch_finished,
                }
                self._save_or_update_checkpoint(save_dict, self.cfg.info.work_dir, epoch_finished, 'best')
    
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
        LoggerMisc.logging(self.loggers, 'val_epoch', self.last_val_metrics, self.trained_iters)
        
        self._save_best_only_model_checkpoint()
    
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
            with self.train_autocast():
                outputs = self.model(inputs)
                loss, metrics_dict = self.criterion(outputs, targets)
        else:
            with torch.no_grad():
                with self.val_autocast():
                    outputs = self.model(inputs)
                    loss, metrics_dict = self.criterion(outputs, targets)
                    
                    if self.ema_container is not None:
                        ema_outputs = self.ema_container(inputs)
                        _, ema_metrics_dict = self.ema_criterion(ema_outputs, targets)
                        outputs.update(LoggerMisc.set_dict_key_prefix(ema_outputs, 'ema_'))
                        metrics_dict.update(ema_metrics_dict)
            
        return outputs, loss, metrics_dict
    
    def _backward_and_step(self, loss: torch.Tensor):
        grad_norm = None
        def _backward():
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
        def _optimize():
            grad_norm = None
            if self.scaler is not None:
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()
            self.optimizer.zero_grad()
            if self.ema_container is not None:
                self.ema_container.update()
            return grad_norm
        
        if not math.isfinite(loss) and self.scaler is None:
            LoggerMisc.get_wandb_pid(kill_all=True)
            raise ValueError(f'Rank {DistMisc.get_rank()}: Loss is {loss}, stopping training.')
        
        if self.do_gradient_accumulation:
            loss /= self.gradient_accumulation_steps  # Assume that all losses are mean-reduction. (Otherwise meaningless)
            self.step_count += 1
        
        _backward()
        
        # if self.do_gradient_accumulation and (self.trained_iters % self.train_len != 0):  # not the last iter of the epoch
        if self.do_gradient_accumulation:  # just drop the last few steps if not divisible
            if self.step_count % self.gradient_accumulation_steps == 0:
                grad_norm = _optimize()
                self.step_count = 0
        else:
            grad_norm = _optimize()
        
        self.lr_scheduler.step()  # update special lr_scheduler after each iter
        self.trained_iters += 1
        return grad_norm
        
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
            'loss': ValueMetric(high_prior=True),
            'grad_norm': ValueMetric(high_prior=True, no_sync=True),
            **{lr_group: ValueMetric(format='{value:.2e}', final_format='[{min:.2e}, {max:.2e}]', low_prior=True, no_sync=True) for lr_group in self.lr_groups.keys()},
            'epoch': ValueMetric(window_size=1, no_print=True, no_sync=True),
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
            
            grad_norm = self._backward_and_step(loss)
            
            mlogger.update_metrics(
                epoch=self.trained_iters / self.train_len,
            )
            if grad_norm is not None:
                mlogger.update_metrics(
                    grad_norm=grad_norm,
                )
            
            if cfg.info.iter_log_freq > 0:
                if self.trained_iters % (cfg.info.iter_log_freq * cfg.trainer.grad_accumulation) == 0:
                    LoggerMisc.logging(loggers, 'train_iter', mlogger.output_dict(no_avg_list=['all']), self.trained_iters)
            
            if first_iter:
                first_iter = False
                self._after_first_train_iter()
        
        mlogger.add_epoch_metrics(**self.criterion.forward_epoch_metrics())
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
        mlogger.add_metrics([{'loss': ValueMetric(high_prior=True)}])
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
        
        mlogger.add_epoch_metrics(**self.criterion.forward_epoch_metrics())
        if hasattr(self, 'ema_criterion'):
            mlogger.add_epoch_metrics(**self.ema_criterion.forward_epoch_metrics())
        self.last_val_metrics = mlogger.output_dict(sync=self.dist_eval, final_print=True)
    
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
