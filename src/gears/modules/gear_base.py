import math
import os
import time
from argparse import Namespace
from glob import glob

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from src.criterions import CriterionBase
from src.models import ModelBase
from src.utils.misc import *
from src.utils.register import Register

trainer_register = Register('trainer')
tester_register = Register('tester')

class TrainerBase:
    def __init__(
        self,
        cfg: Namespace,
        loggers: Namespace,
        model: ModelBase,
        criterion: CriterionBase,
        train_loader: DataLoader,
        val_loader: DataLoader,
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
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        self.device = device
        self.start_epoch = 1
        self.epoch = self.start_epoch - 1
        self.train_iters = 0
        self.train_outputs = {}
        self.metrics = {}
        self.best_metrics = {}
        self.train_pbar = None
        self.val_pbar = None
        
        assert self.cfg.trainer.grad_accumulation > 0 and isinstance(self.cfg.trainer.grad_accumulation, int), 'grad_accumulation should be a positive integer.'
        self.gradient_accumulation_steps = self.cfg.trainer.grad_accumulation
        self.do_gradient_accumulation = self.gradient_accumulation_steps > 1
        self.max_grad_norm = self.cfg.trainer.max_grad_norm
        self.train_len = len(self.train_loader)
        self.val_len = len(self.val_loader)
          
        self.nn_module_list = [self.model, self.criterion]
        self.is_train = True
        
        self.breath_time = self.cfg.trainer.trainer_breath_time  # XXX: avoid cpu being too busy
    
    def _get_pbar(self):
        # called in "before_all_epochs"
        if DistMisc.is_main_process():
            epoch_finished = self.start_epoch - 1
            train_pbar = LoggerMisc.MultiTQDM(
                total=self.cfg.trainer.epochs * self.train_len if self.cfg.info.global_tqdm else self.train_len,
                dynamic_ncols=True,
                colour='green',
                position=0,
                maxinterval=math.inf,
                initial=epoch_finished * self.train_len,
            )
            train_pbar.set_description_str('Train')
            print('')
            val_pbar = LoggerMisc.MultiTQDM(
                total=self.cfg.trainer.epochs * self.val_len if self.cfg.info.global_tqdm else self.val_len,
                dynamic_ncols=True,
                colour='green',
                position=0,
                maxinterval=math.inf,
                initial=epoch_finished * self.val_len,
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
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if self.cfg.env.amp:
                self.scaler.load_state_dict(checkpoint['scaler'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_metrics = checkpoint.get('best_metrics', {})
            self.metrics = checkpoint.get('last_metrics', {})
            self.train_iters = checkpoint['epoch'] * self.train_len
            self.epoch = self.start_epoch - 1
        else:
            print(LoggerMisc.block_wrapper('New trainer.', '>'))
        print(f'Start from epoch: {self.start_epoch}')
        return self.cfg.trainer.resume
            
    def _save_checkpoint(self):
        # called in "after_validation"
        if DistMisc.is_main_process():
            epoch_finished = self.epoch
            self.best_metrics, save_flag = self.criterion.choose_best(
                self.metrics, self.best_metrics
            )

            save_files = {
                'model': self.model_without_ddp.state_dict(),
                'best_metrics': self.best_metrics,
                'epoch': epoch_finished,
            }

            if save_flag:
                best = glob(os.path.join(self.cfg.info.work_dir, 'checkpoint_best_epoch_*.pth'))
                assert len(best) <= 1
                if len(best) == 1:
                    torch.save(save_files, best[0])
                    os.rename(best[0], os.path.join(self.cfg.info.work_dir, f'checkpoint_best_epoch_{epoch_finished}.pth'))
                else:
                    torch.save(save_files, os.path.join(self.cfg.info.work_dir, f'checkpoint_best_epoch_{epoch_finished}.pth'))

            if (self.epoch + 1) % self.cfg.trainer.save_interval == 0:
                save_files.update({
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                    'last_metrics': self.metrics,
                })
                if self.cfg.env.amp and self.cfg.env.device:
                    save_files.update({
                        'scaler': self.scaler.state_dict()
                    })
                last = glob(os.path.join(self.cfg.info.work_dir, 'checkpoint_last_epoch_*.pth'))
                assert len(last) <= 1
                if len(last) == 1:
                    torch.save(save_files, last[0])
                    os.rename(last[0], os.path.join(self.cfg.info.work_dir, f'checkpoint_last_epoch_{epoch_finished}.pth'))
                else:
                    torch.save(save_files, os.path.join(self.cfg.info.work_dir, f'checkpoint_last_epoch_{epoch_finished}.pth'))
                    
    def _train_mode(self):
        # called in "before_one_epoch"
        for nn_module in self.nn_module_list:
            nn_module.train()
        self.is_train = True
            
    def _eval_mode(self):
        # called in "after_training_before_validation"
        for nn_module in self.nn_module_list:
            nn_module.eval()
        self.is_train = False
                    
    def forward(self, batch: dict):
        time.sleep(self.breath_time)
        
        batch: dict = TensorMisc.to(batch, self.device, non_blocking=self.cfg.env.pin_memory)
        inputs: dict = batch['inputs']
        targets: dict = batch['targets']
        
        if self.is_train:
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                outputs = self.model(**inputs)
                loss, metrics_dict = self.criterion(outputs, targets)
            self.train_iters += 1
        else:
            with torch.no_grad():
                outputs = self.model(**inputs)
                loss, metrics_dict = self.criterion(outputs, targets)
            
        return outputs, loss, metrics_dict
    
    def backward_and_step(self, loss: torch.tensor):       
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
        
        if not math.isfinite(loss):
            LoggerMisc.get_wandb_pid(kill_all=True)
            raise ValueError(f'Rank {DistMisc.get_rank()}: Loss is {loss}, stopping training.')
        
        if self.do_gradient_accumulation:
            loss /= self.gradient_accumulation_steps  # Assume that all losses are mean-reduction. (Otherwise meaningless)
            self.step_count += 1
        
        _backward()
        
        if self.do_gradient_accumulation and (self.train_iters % self.train_len != 0):
            if self.step_count % self.gradient_accumulation_steps == 0:
                _optimize()
                self.step_count = 0
        else:
            _optimize()
        
        self.lr_scheduler.step()  # update special lr_scheduler after each iter
    
    @property
    def epoch_loop(self):
        return range(self.start_epoch, self.cfg.trainer.epochs + 1)
    
    @property
    def lr_groups(self):
        return {'lr_' + param_group['group_name']: param_group['lr'] for param_group in self.optimizer.param_groups}
    
    @property
    def wd_groups(self):
        return {'wd_' + param_group['group_name']: param_group['weight_decay'] for param_group in self.optimizer.param_groups}
    
    def before_all_epochs(self, **kwargs):
        self._resume_training()
        ModelMisc.show_model_info(self.cfg, self)
        self._get_pbar()
    
    def before_one_epoch(self, **kwargs):
        self.epoch += 1
        if self.cfg.env.distributed:
            # shuffle data for each epoch (here needs epoch start from 0)
            self.train_loader.sampler_set_epoch(self.epoch - 1)  
        
        dist.barrier()

        if DistMisc.is_main_process():
            if self.cfg.info.global_tqdm:
                self.train_pbar.unpause()
            else :
                self.train_pbar.reset()
                self.val_pbar.reset()
          
        self.step_count = 0
        self.optimizer.zero_grad()
        self._train_mode()

    def after_training_before_validation(self, **kwargs):
        LoggerMisc.logging(self.loggers, 'train_epoch', self.train_outputs, self.train_iters)
        
        dist.barrier()

        if DistMisc.is_main_process():
            self.val_pbar.unpause()
            
        self._eval_mode()

    def after_validation(self, **kwargs):
        LoggerMisc.logging(self.loggers, 'val_epoch', self.metrics, self.train_iters)
        
        self._save_checkpoint()
    
    def after_all_epochs(self, **kwargs):
        dist.barrier()

        if DistMisc.is_main_process():
            self.train_pbar.close()
            self.val_pbar.close()


class TesterBase:
    def __init__(
        self,
        cfg: Namespace,
        loggers: Namespace,
        model: ModelBase,
        criterion: CriterionBase,
        test_loader: DataLoader,
        device: torch.device,
        ) -> None:
        super().__init__()
        self.cfg = cfg
        self.loggers = loggers
        self.model = model
        self.model_without_ddp = model.module if cfg.env.distributed else model
        self.criterion = criterion
        self.test_loader = test_loader
        self.device = device
        self.metrics = {}
        self.test_pbar = None
        
        self.test_len = len(self.test_loader)
        
        self.nn_module_list = [self.model, self.criterion]
            
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
        self.model_without_ddp.load_state_dict(checkpoint['model'])
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
        # self.is_train = False
        
    def forward(self, batch: dict):
        time.sleep(self.breath_time)
        
        batch: dict = TensorMisc.to(batch, self.device, non_blocking=self.cfg.env.pin_memory)
        inputs: dict = batch['inputs']
        targets: dict = batch['targets']
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            loss, metrics_dict = self.criterion(outputs, targets, infer_mode=True)
            
        return outputs, loss, metrics_dict

    def before_inference(self, **kwargs):
        self._load_model()
        self._get_pbar()
        
        self._eval_mode()

    def after_inference(self, **kwargs):
        LoggerMisc.logging(self.loggers,  'infer', self.metrics, None)
        
        if DistMisc.is_main_process():          
            self.test_pbar.close()

        