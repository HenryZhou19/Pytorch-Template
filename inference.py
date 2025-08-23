from src.gears import GearManager
from src.utils.misc import *


def infer_portal(infer_cfg):
    # set start_time and broadcast it to all ranks
    PortalMisc.set_and_broadcast_start_time(infer_cfg, 'infer_start_time')
    
    # combine train(read) inference(input) configs
    cfg = PortalMisc.combine_train_infer_configs(infer_cfg, use_train_seed=True)
    
    # interrupt handler
    PortalMisc.interrupt_handler(cfg)
    
    # special config adjustment (debug and work_dir for inference)
    PortalMisc.special_config_adjustment(cfg)
    
    # seed everything
    PortalMisc.seed_everything(cfg)
    
    # save configs to work_dir as .yaml file (and save current project files, print config to CLI if needed)
    PortalMisc.save_configs(cfg)
    
    # init loggers (wandb/tensorboard and local:log_file)
    loggers = PortalMisc.init_loggers(cfg)
    
    # main tester
    tester = GearManager(cfg, loggers).build_tester()
    tester.run()
    
    # end everything
    PortalMisc.end_everything(cfg, loggers)
    
    return cfg


if __name__ == '__main__':
    infer_cfg = ConfigMisc.get_configs()
    
    # init distributed mode
    DistMisc.init_distributed_mode(infer_cfg)
    
    SweepMisc.init_sweep_mode(infer_cfg, infer_portal)
    
    DistMisc.destroy_process_group()
