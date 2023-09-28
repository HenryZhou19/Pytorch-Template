import torch

from src.criterions import CriterionManager
from src.datasets import DataManager
from src.engine import evaluate, train_one_epoch
from src.gears import Trainer
from src.models import ModelManager
from src.utils.misc import (ConfigMisc, DistMisc, ModelMisc, OptimizerMisc,
                            PortalMisc, SchudulerMisc, SweepMisc, TimeMisc)


def train_run(cfg):

    # prepare for model, postprocessor
    model_manager = ModelManager(cfg)
    model_without_ddp = model_manager.build_model()
    # postprocessor = model_manager.build_postprocessor()
    
    # prepare for criterion
    criterion_manager = CriterionManager(cfg)
    loss_criterion, metric_criterion = criterion_manager.build_criterion()

    # prepare for data
    data_manager = DataManager(cfg)
    train_loader = data_manager.build_dataset(split='train', shuffle=True)
    val_loader = data_manager.build_dataset(split='val')

    # prepare for optimizer
    optimizer = OptimizerMisc.get_optimizer(cfg, model_without_ddp)
    
    # prepare for cuda auto mixed precision(amp)
    scaler = torch.cuda.amp.GradScaler() if cfg.env.amp and cfg.env.device=='cuda' else None

    # prepare for lr_scheduler
    lr_scheduler = SchudulerMisc.get_warmup_lr_scheduler(cfg, optimizer, scaler, train_loader)
    
    # print model info
    # ModelMisc.print_model_info(cfg, model_without_ddp, 'model_structure', 'torchinfo')
    ModelMisc.print_model_info_with_torchinfo(cfg, model_without_ddp, train_loader, ['input_size', 'output_size', 'num_params', 'params_percent', 'kernel_size', 'mult_adds', 'trainable',])
    
    # model wrapper
    model = ModelMisc.ddp_wrapper(cfg, model_without_ddp)

    # trainer_status as the global status for inputs and outputs of each epoch
    trainer_status = {
        'model': model,
        'model_without_ddp': model_without_ddp,
        'loss_criterion': loss_criterion,
        'metric_criterion': metric_criterion,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler,
        'scaler': scaler,
        'device': model_manager.device,  # torch.device
        'start_epoch': 1,
        'epoch': 0, 
        'train_iters': 0,
        'train_outputs': {},
        'metrics': {},
        'best_metrics': {},
        'train_pbar': None,
        'val_pbar': None,
    }

    # prepare for resumed training if needed
    trainer_status = Trainer.resume_training(cfg, trainer_status)
    
    # prepare for progress bar
    trainer_status = Trainer.get_pbar(cfg, trainer_status)
    
    for epoch in range(trainer_status['start_epoch'], cfg.trainer.epochs + 1):
        Trainer.before_one_epoch(cfg, trainer_status, epoch=epoch)

        trainer_status['train_outputs'] = train_one_epoch(cfg, trainer_status)
        
        Trainer.after_training_before_validation(cfg, trainer_status)

        if DistMisc.is_main_process() or cfg.trainer.dist_eval:
            trainer_status['metrics'] = evaluate(cfg, trainer_status)

        Trainer.after_validation(cfg, trainer_status)

        if cfg.special.debug:
            PortalMisc.end_everything(cfg, force=True)
            
    Trainer.after_all_epochs(cfg, trainer_status)


def train_portal(cfg):
    setattr(cfg.info, 'start_time', TimeMisc.get_time_str())
    
    # special config adjustment(debug)
    PortalMisc.special_config_adjustment(cfg)
    
    # resume training or new one (makedirs & write configs to file)
    PortalMisc.resume_or_new_train_dir(cfg)

    # seed everything
    PortalMisc.seed_everything(cfg)

    # save configs to work_dir as .yaml file (and save current project files if needed)
    PortalMisc.save_configs(cfg)

    # choose whether to print configs of each rank
    PortalMisc.print_config(cfg, force_all_rank=False)

    # init loggers(wandb and local:log_file)
    PortalMisc.init_loggers(cfg)

    # interrupt handler
    PortalMisc.interrupt_handler(cfg)
    
    # main trainer
    train_run(cfg)

    # end everything
    PortalMisc.end_everything(cfg, end_with_printed_cfg=True)

            
if __name__ == '__main__':
    cfg = ConfigMisc.get_configs_from_sacred(main_config='./configs/train.yaml')
    assert hasattr(cfg, 'info'), 'cfg.info not found'
    
    # init distributed mode
    DistMisc.init_distributed_mode(cfg)
    
    SweepMisc.init_sweep_mode(cfg, train_portal)
