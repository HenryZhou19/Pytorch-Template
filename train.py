from src.criterions import CriterionManager
from src.datasets import DataManager
from src.engine import evaluate, train_one_epoch
from src.gears import GearManager
from src.models import ModelManager
from src.utils.misc import (ConfigMisc, DistMisc, ModelMisc, PortalMisc,
                            SweepMisc, TimeMisc)
from src.utils.optimizer import OptimizerUtils, SchedulerUtils


def train_run(cfg, loggers):

    # prepare for data
    data_manager = DataManager(cfg, loggers)
    train_loader = data_manager.build_dataloader(split='train')
    val_loader = data_manager.build_dataloader(split='val')
    
    # prepare for model, postprocessor
    model_manager = ModelManager(cfg, loggers)
    model_without_ddp = model_manager.build_model()
    # postprocessor = model_manager.build_postprocessor()
    
    # prepare for criterion
    criterion_manager = CriterionManager(cfg, loggers)
    criterion = criterion_manager.build_criterion()

    # prepare for optimizer and scaler (cuda auto mixed precision(amp)) if needed
    optimizer, scaler = OptimizerUtils.get_optimizer(cfg, model_without_ddp)

    # prepare for lr_scheduler
    lr_scheduler = SchedulerUtils.get_warmup_lr_scheduler(cfg, optimizer, scaler, train_loader)
    
    # model wrapper
    model = ModelMisc.ddp_wrapper(cfg, model_without_ddp)

    # get Trainer instance
    trainer = GearManager(cfg, loggers).build_trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        device=model_manager.device,
        )

    # prepare for 1. resumed training if needed; 2. show model information; 3. progress bar;
    trainer.before_all_epochs()
    
    for _ in trainer.epoch_loop:
        trainer.before_one_epoch()

        train_one_epoch(trainer)
        
        trainer.after_training_before_validation()

        if cfg.trainer.dist_eval or DistMisc.is_main_process():
            evaluate(trainer)

        trainer.after_validation()

        if cfg.special.debug == 'one_epoch':
            PortalMisc.end_everything(cfg, loggers, force=True)
            
    trainer.after_all_epochs()


def train_portal(cfg):
    assert hasattr(cfg, 'info'), 'config field "cfg.info" not found'
    setattr(cfg.info, 'start_time', TimeMisc.get_time_str())
    
    # interrupt handler
    PortalMisc.interrupt_handler(cfg)
    
    # special config adjustment (debug)
    PortalMisc.special_config_adjustment(cfg)
    
    # resume training or new one (makedirs & write configs to file)
    PortalMisc.resume_or_new_train_dir(cfg)

    # seed everything
    PortalMisc.seed_everything(cfg)

    # save configs to work_dir as .yaml file (and save current project files if needed)
    PortalMisc.save_configs(cfg)

    # choose whether to print configs of each rank
    PortalMisc.print_config(cfg, force_all_rank=False)

    # init loggers (wandb/tensorboard and local:log_file)
    loggers = PortalMisc.init_loggers(cfg)
    
    # main trainer
    train_run(cfg, loggers)

    # end everything
    PortalMisc.end_everything(cfg, loggers, end_with_printed_cfg=True)

            
if __name__ == '__main__':
    cfg = ConfigMisc.get_configs(config_dir='./configs/')
    
    # init distributed mode
    DistMisc.init_distributed_mode(cfg)
    
    SweepMisc.init_sweep_mode(cfg, train_portal)
