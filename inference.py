from src.criterions import CriterionManager
from src.datasets import DataManager
from src.engine import test
from src.gears import GearManager
from src.models import ModelManager
from src.utils.misc import (ConfigMisc, DistMisc, ModelMisc, PortalMisc,
                            SweepMisc, TimeMisc)


def test_run(cfg,loggers):

    # prepare for data
    data_manager = DataManager(cfg, loggers)
    test_loader = data_manager.build_dataloader(split='test')
    
    # prepare for model, postprocessor
    model_manager = ModelManager(cfg, loggers)
    model_without_ddp = model_manager.build_model()
    # postprocessor = model_manager.build_postprocessor()
    
    # prepare for criterion
    criterion_manager = CriterionManager(cfg, loggers)
    criterion = criterion_manager.build_criterion()
    
    # model wrapper
    model = ModelMisc.ddp_wrapper(cfg, model_without_ddp)

    # get Tester instance
    tester = GearManager(cfg, loggers).build_tester(
        model=model,
        criterion=criterion,
        test_loader=test_loader,
        device=model_manager.device,
        )

    # prepare for 1. loading model; 2. progress bar
    tester.before_inference()

    test(tester)

    tester.after_inference()


def infer_portal(infer_cfg):
    assert hasattr(infer_cfg, 'info'), 'config field "infer_cfg.info" not found'
    setattr(infer_cfg.info, 'infer_start_time', TimeMisc.get_time_str())
    
    # combine train(read) inference(input) configs
    cfg = PortalMisc.combine_train_infer_configs(infer_cfg, use_train_seed=True)
    
    # special config adjustment (debug and work_dir for inference)
    PortalMisc.special_config_adjustment(cfg)
    
    # seed everything
    PortalMisc.seed_everything(cfg)

    # save configs to work_dir as .yaml file (and save current project files if needed)
    PortalMisc.save_configs(cfg)

    # choose whether to print configs of each rank
    PortalMisc.print_config(cfg, force_all_rank=False)

    # init loggers (wandb/tensorboard and local:log_file)
    loggers = PortalMisc.init_loggers(cfg)
    
    # interrupt handler
    PortalMisc.interrupt_handler(cfg)

    # main tester
    test_run(cfg, loggers)
    
    # end everything
    PortalMisc.end_everything(cfg, loggers, end_with_printed_cfg=True)


if __name__ == '__main__':
    infer_cfg = ConfigMisc.get_configs(config_dir='./configs/')

    # init distributed mode
    DistMisc.init_distributed_mode(infer_cfg)
    
    SweepMisc.init_sweep_mode(infer_cfg, infer_portal)
