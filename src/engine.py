from src.gears import TesterBase, TrainerBase
from src.utils.misc import LoggerMisc
from src.utils.progress_logger import MetricLogger
from src.utils.progress_logger import SmoothedValue as SV


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
        'loss': SV(prior=True),
        **{lr_group: SV(format='{value:.2e}', final_format='[{min:.2e}, {max:.2e}]', no_sync=True) for lr_group in trainer.lr_groups.keys()},
        'epoch': SV(window_size=1, no_print=True, no_sync=True)
        }])
    for batch in mlogger.log_every(trainer.train_loader):
        
        _, loss, metrics_dict = trainer.forward(batch)
        
        mlogger.update(
            **trainer.lr_groups,
            epoch=trainer.train_iters / trainer.train_len,
            loss=loss,
            **metrics_dict,
        )
        
        if cfg.info.iter_log_freq > 0:
            if trainer.train_iters % (cfg.info.iter_log_freq * cfg.trainer.grad_accumulation) == 0:
                LoggerMisc.logging(loggers, 'train_iter', mlogger.output_dict(no_avg_list=['all']), int(trainer.train_iters / cfg.trainer.grad_accumulation))

        trainer.backward_and_step(loss)
    
    trainer.train_outputs = mlogger.output_dict(no_avg_list=[*trainer.lr_groups.keys(), 'epoch'], sync=True, final_print=True)
    trainer.train_outputs.update(trainer.criterion.get_epoch_metrics())


def evaluate(trainer: TrainerBase):
    cfg = trainer.cfg
    
    mlogger = MetricLogger(
        cfg=cfg,
        loggers=trainer.loggers,
        pbar=trainer.val_pbar,  
        header='Eval ',
        epoch_str=f'epoch: [{trainer.epoch}/{cfg.trainer.epochs}]',
        )
    mlogger.add_meters([{'loss': SV(prior=True)}])
    for batch in mlogger.log_every(trainer.val_loader):
        
        _, loss, metrics_dict = trainer.forward(batch)

        mlogger.update(
            loss=loss,
            **metrics_dict,
        )
        
    trainer.metrics = mlogger.output_dict(sync=cfg.trainer.dist_eval, final_print=True)
    trainer.metrics.update(trainer.criterion.get_epoch_metrics())


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
            
        mlogger.update(**metrics_dict)
        
    tester.metrics = mlogger.output_dict(sync=True, final_print=True)
    tester.metrics.update(tester.criterion.get_epoch_metrics())
