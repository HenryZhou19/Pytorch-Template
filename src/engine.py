import math

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.loss import LossBase
from src.utils.metric import MetricBase
from src.utils.misc import DistMisc, LoggerMisc
from src.utils.progress_logger import MetricLogger
from src.utils.progress_logger import SmoothedValue as SV


def train_one_epoch(cfg, trainer_status):
    model: nn.Module = trainer_status['model']
    loss_criterion: LossBase = trainer_status['loss_criterion']
    loader: DataLoader = trainer_status['train_loader']
    epoch: int = trainer_status['epoch']  # start from 1
    device: torch.device = trainer_status['device']
    optimizer: torch.optim.Optimizer = trainer_status['optimizer']
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = trainer_status['lr_scheduler']
    warmup_lr_scheduler: torch.optim.lr_scheduler._LRScheduler = trainer_status['warmup_lr_scheduler']
    scaler: torch.cuda.amp.GradScaler = trainer_status['scaler']
    pbar: tqdm = trainer_status['train_pbar']

    model.train()
    loss_criterion.train()
        
    logger = MetricLogger(
        log_file=cfg.info.log_file,
        print_freq=cfg.info.cli_log_freq,
        debug=cfg.special.debug,
        pbar=pbar,
        global_tqdm=cfg.info.global_tqdm,
        header='Train',
        epoch_str='epoch: [{}/{}]'.format(epoch, cfg.trainer.epochs),
        )
    logger.add_meters([{
        'loss': SV(prior=True),
        'lr': SV(window_size=1, fmt='{value:.2e}', final_fmt='[{min:.2e}, {max:.2e}]', no_sync=True),
        'epoch': SV(window_size=1, no_print=True, no_sync=True)
        }])
    for batch in logger.log_every(loader):
        inputs, targets = batch['inputs'].to(device), batch['targets'].to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(inputs)
            loss, loss_dict = loss_criterion(outputs, targets)

        if not math.isfinite(loss):
            raise ValueError(f'Rank {DistMisc.get_rank()}: Loss is {loss}, stopping training')
            
        trainer_status['train_iters'] += 1 

        logger.update(
            loss=loss,
            lr=optimizer.param_groups[0]['lr'],
            epoch=trainer_status['train_iters'] / len(loader), 
            **loss_dict)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if cfg.trainer.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=cfg.trainer.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.trainer.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=cfg.trainer.max_grad_norm)
            optimizer.step()

        if warmup_lr_scheduler is not None and epoch == 1:  # only in the first epoch
            warmup_lr_scheduler.step()  # update warmup_lr_scheduler after each iter (sub scheduler)
            
        if cfg.info.wandb_log_freq > 0:
            if trainer_status['train_iters'] % cfg.info.wandb_log_freq == 0:
                output_dict = logger.output_dict(no_avg_list=['all'])
                LoggerMisc.wandb_log(cfg,  'train_iter', output_dict, trainer_status['train_iters'])
            
    if lr_scheduler is not None:
            lr_scheduler.step()  # update lr_scheduler after each epoch (main scheduler)

    return logger.output_dict(no_avg_list=['lr', 'epoch'], sync=True, final_print=True)


def evaluate(cfg, trainer_status):
    model: nn.Module = trainer_status['model']
    loss_criterion: LossBase = trainer_status['loss_criterion']
    metric_criterion: MetricBase = trainer_status['metric_criterion']
    loader: DataLoader = trainer_status['val_loader']
    epoch: int = trainer_status['epoch']
    device: torch.device = trainer_status['device']
    pbar: tqdm = trainer_status['val_pbar']

    model.eval()
    loss_criterion.eval()

    logger = MetricLogger(
        log_file=cfg.info.log_file,
        print_freq=cfg.info.cli_log_freq,
        debug=cfg.special.debug,
        pbar=pbar,
        global_tqdm=cfg.info.global_tqdm,
        header='Eval ',
        epoch_str='epoch: [{}/{}]'.format(epoch, cfg.trainer.epochs),
        )
    logger.add_meters([{'loss': SV(prior=True)}])
    for batch in logger.log_every(loader):
        inputs, targets = batch['inputs'].to(device), batch['targets'].to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss, loss_dict = loss_criterion(outputs, targets)

            metrics, *_ = metric_criterion.get_metrics(outputs, targets)
            logger.update(**metrics)

        logger.update(
            loss=loss,
            **loss_dict,
        )
        
    sync = cfg.trainer.dist_eval
    return logger.output_dict(sync=sync, final_print=True)


def test(cfg, tester_status):
    model: nn.Module = tester_status['model']
    metric_criterion: MetricBase = tester_status['metric_criterion']
    loader: DataLoader = tester_status['test_loader']
    device: torch.device = tester_status['device']
    pbar: tqdm = tester_status['test_pbar']
    
    model.eval()

    logger = MetricLogger(
        log_file=cfg.info.log_file,
        print_freq=cfg.info.cli_log_freq,
        debug=cfg.special.debug,
        pbar=pbar,
        header='Test',
        )
    for batch in logger.log_every(loader):
        inputs, targets = batch['inputs'].to(device), batch['targets'].to(device)
        with torch.no_grad():
            outputs = model(inputs)

        metrics, *_= metric_criterion.get_metrics(outputs, targets)

        logger.update(**metrics)
            
    return logger.output_dict(sync=True, final_print=True)
