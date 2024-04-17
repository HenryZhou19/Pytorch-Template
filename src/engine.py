from src.gears import TesterBase, TrainerBase
from src.utils.misc import LoggerMisc
from src.utils.progress_logger import MetricLogger
from src.utils.progress_logger import ValueMeter as VM


def train_one_epoch(trainer: TrainerBase):
    cfg = trainer.cfg
    loggers = trainer.loggers
    
    mlogger = MetricLogger(
        cfg=cfg,
        loggers=loggers,
        pbar=trainer.train_pbar,  
        header='Train',
        epoch_str=f'epoch: [{trainer.epoch}/{cfg.trainer.epochs}]',
        )
    mlogger.add_meters([{
        'loss': VM(prior=True),
        **{lr_group: VM(format='{value:.2e}', final_format='[{min:.2e}, {max:.2e}]', no_sync=True) for lr_group in trainer.lr_groups.keys()},
        'epoch': VM(window_size=1, no_print=True, no_sync=True)
        }])
    for batch in mlogger.log_every(trainer.train_loader):
        
        _, loss, metrics_dict = trainer.forward(batch)
        
        mlogger.update_meters(
            sample_count=batch['batch_size'],
            **trainer.lr_groups,
            loss=loss,
            **metrics_dict,
        )
        
        trainer.backward_and_step(loss)
        
        mlogger.update_meters(
            epoch=trainer.trained_iters / trainer.train_len,
        )
        
        if cfg.info.iter_log_freq > 0:
            if trainer.trained_iters % (cfg.info.iter_log_freq * cfg.trainer.grad_accumulation) == 0:
                LoggerMisc.logging(loggers, 'train_iter', mlogger.output_dict(no_avg_list=['all']), int(trainer.trained_iters / cfg.trainer.grad_accumulation))
    
    mlogger.add_epoch_meters(**trainer.criterion.get_epoch_metrics_and_reset())
    trainer.train_outputs = mlogger.output_dict(no_avg_list=[*trainer.lr_groups.keys(), 'epoch'], sync=True, final_print=True)


def evaluate(trainer: TrainerBase):
    cfg = trainer.cfg
    
    mlogger = MetricLogger(
        cfg=cfg,
        loggers=trainer.loggers,
        pbar=trainer.val_pbar,  
        header='Eval ',
        epoch_str=f'epoch: [{trainer.epoch}/{cfg.trainer.epochs}]',
        )
    mlogger.add_meters([{'loss': VM(prior=True)}])
    for batch in mlogger.log_every(trainer.val_loader):
        
        _, loss, metrics_dict = trainer.forward(batch)

        mlogger.update_meters(
            sample_count=batch['batch_size'],
            loss=loss,
            **metrics_dict,
        )
    
    mlogger.add_epoch_meters(**trainer.criterion.get_epoch_metrics_and_reset())
    if hasattr(trainer, 'ema_criterion'):
        ema_epoch_metrics = {}
        raw_epoch_metrics = trainer.ema_criterion.get_epoch_metrics_and_reset()
        for k, v in raw_epoch_metrics.items():
            ema_epoch_metrics[f'ema_{k}'] = v
        mlogger.add_epoch_meters(**ema_epoch_metrics)
    trainer.metrics = mlogger.output_dict(sync=cfg.trainer.dist_eval, final_print=True)


def test(tester: TesterBase):
    cfg = tester.cfg

    mlogger = MetricLogger(
        cfg=cfg,
        loggers=tester.loggers,
        pbar=tester.test_pbar,  
        header='Test',
        )
    for batch in mlogger.log_every(tester.test_loader):
        
        outputs, _, metrics_dict = tester.forward(batch)
            
        mlogger.update_meters(
            sample_count=batch['batch_size'],
            **metrics_dict,
            )
        
    mlogger.add_epoch_meters(**tester.criterion.get_epoch_metrics_and_reset())
    if hasattr(tester, 'ema_criterion'):
        ema_epoch_metrics = {}
        raw_epoch_metrics = tester.ema_criterion.get_epoch_metrics_and_reset()
        for k, v in raw_epoch_metrics.items():
            ema_epoch_metrics[f'ema_{k}'] = v
        mlogger.add_epoch_meters(**ema_epoch_metrics)
    tester.metrics = mlogger.output_dict(sync=True, final_print=True)
