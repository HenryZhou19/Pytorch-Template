from src.gears import GearManager
from src.utils.misc import *


def train_portal(cfg):
    # set start_time and broadcast it to all ranks
    PortalMisc.set_and_broadcast_start_time(cfg, 'start_time')
    
    # interrupt handler
    PortalMisc.interrupt_handler(cfg)
    
    # special config adjustment (debug)
    PortalMisc.special_config_adjustment(cfg)
    
    # resume training or new one (makedirs & write configs to file)
    PortalMisc.resume_or_new_train_dir(cfg)
    
    # seed everything
    PortalMisc.seed_everything(cfg)
    
    # save configs to work_dir as .yaml file (and save current project files, print config to CLI if needed)
    PortalMisc.save_configs(cfg)
    
    # init loggers (wandb/tensorboard and local:log_file)
    loggers = PortalMisc.init_loggers(cfg)
    
    # main trainer
    trainer = GearManager(cfg, loggers).build_trainer()
    trainer.run()
    
    # end everything
    PortalMisc.end_everything(cfg, loggers)


if __name__ == '__main__':
    cfg = ConfigMisc.get_configs()
    
    # init distributed mode
    DistMisc.init_distributed_mode(cfg)
    
    SweepMisc.init_sweep_mode(cfg, train_portal)
    
    DistMisc.destroy_process_group()
