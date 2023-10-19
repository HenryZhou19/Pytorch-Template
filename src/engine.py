from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.criterions.simple_criterion import CriterionBase
from src.utils.misc import LoggerMisc, TensorMisc, TrainerMisc
from src.utils.progress_logger import MetricLogger
from src.utils.progress_logger import SmoothedValue as SV


def train_one_epoch(cfg, trainer_status):
    model: torch.nn.Module = trainer_status['model']
    criterion: CriterionBase = trainer_status['criterion']
    loader: DataLoader = trainer_status['train_loader']
    epoch: int = trainer_status['epoch']  # start from 1
    device: torch.device = trainer_status['device']
    optimizer: torch.optim.Optimizer = trainer_status['optimizer']
    scaler: torch.cuda.amp.GradScaler = trainer_status['scaler']
    pbar: LoggerMisc.MultiTQDM = trainer_status['train_pbar']

    model.train()
    criterion.train()
    backward_and_step = TrainerMisc.BackwardAndStep(cfg, trainer_status)
    
    logger = MetricLogger(
        cfg=cfg,
        pbar=pbar,  
        header='Train',
        epoch_str='epoch: [{}/{}]'.format(epoch, cfg.trainer.epochs),
        )
    logger.add_meters([{
        'loss': SV(prior=True),
        'lr': SV(fmt='{value:.2e}', final_fmt='[{min:.2e}, {max:.2e}]', no_sync=True),
        'epoch': SV(window_size=1, no_print=True, no_sync=True)
        }])
    for batch in logger.log_every(loader):
        inputs: Dict = TensorMisc.to(batch['inputs'], device)
        targets: Dict = TensorMisc.to(batch['targets'], device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(**inputs)
            loss, metrics_dict = criterion(outputs, targets)
        trainer_status['train_iters'] += 1
        
        logger.update(
            lr=optimizer.param_groups[0]['lr'],  # Assume only one group. TODO: multiple groups lr logging
            epoch=trainer_status['train_iters'] / logger.iter_len,
            loss=loss,
            **metrics_dict,
        )
        
        if cfg.info.wandb_log_freq > 0:
            if trainer_status['train_iters'] % cfg.info.wandb_log_freq == 0:
                LoggerMisc.wandb_log(cfg,  'train_iter', logger.output_dict(no_avg_list=['all']), trainer_status['train_iters'])

        backward_and_step(loss)

    return logger.output_dict(no_avg_list=['lr', 'epoch'], sync=True, final_print=True)


def evaluate(cfg, trainer_status):
    model: torch.nn.Module = trainer_status['model']
    criterion: CriterionBase = trainer_status['criterion']
    loader: DataLoader = trainer_status['val_loader']
    epoch: int = trainer_status['epoch']
    device: torch.device = trainer_status['device']
    pbar: LoggerMisc.MultiTQDM = trainer_status['val_pbar']

    model.eval()
    criterion.eval()
    
    logger = MetricLogger(
        cfg=cfg,
        pbar=pbar,  
        header='Eval ',
        epoch_str='epoch: [{}/{}]'.format(epoch, cfg.trainer.epochs),
        )
    logger.add_meters([{'loss': SV(prior=True)}])
    for batch in logger.log_every(loader):
        inputs: Dict = TensorMisc.to(batch['inputs'], device)
        targets: Dict = TensorMisc.to(batch['targets'], device)
        with torch.no_grad():
            outputs = model(**inputs)
            loss, metrics_dict = criterion(outputs, targets)

        logger.update(
            loss=loss,
            **metrics_dict,
        )      
        
    sync = cfg.trainer.dist_eval
    return logger.output_dict(sync=sync, final_print=True)


def test(cfg, tester_status):
    model: torch.nn.Module = tester_status['model']
    criterion: CriterionBase = tester_status['criterion']
    loader: DataLoader = tester_status['test_loader']
    device: torch.device = tester_status['device']
    pbar: LoggerMisc.MultiTQDM = tester_status['test_pbar']
    
    model.eval()
    criterion.eval()

    logger = MetricLogger(
        cfg=cfg,
        pbar=pbar,  
        header='Test',
        )
    for batch in logger.log_every(loader):
        inputs: Dict = TensorMisc.to(batch['inputs'], device)
        targets: Dict = TensorMisc.to(batch['targets'], device)
        with torch.no_grad():
            outputs = model(**inputs)
            _, metrics_dict = criterion(outputs, targets, test_mode=True)
            
        logger.update(**metrics_dict)
            
    return logger.output_dict(sync=True, final_print=True)
