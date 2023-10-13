from src.criterions import CriterionManager
from src.datasets import DataManager
from src.engine import test
from src.gears import Tester
from src.models import ModelManager
from src.utils.misc import (ConfigMisc, DistMisc, ModelMisc, PortalMisc,
                            SweepMisc, TimeMisc)


def test_run(cfg):

    # prepare for model, postprocessor
    model_manager = ModelManager(cfg)
    model_without_ddp = model_manager.build_model()
    # postprocessor = model_manager.build_postprocessor()
    
    # prepare for criterion
    criterion_manager = CriterionManager(cfg)
    criterion = criterion_manager.build_criterion()

    # prepare for data
    data_manager = DataManager(cfg)
    test_loader = data_manager.build_dataset(split='test')
    
    # model wrapper
    model = ModelMisc.ddp_wrapper(cfg, model_without_ddp)

    tester_status = {
        'model': model,
        'model_without_ddp': model_without_ddp,
        'criterion': criterion,
        'test_loader': test_loader,
        'device': model_manager.device,  # torch.device
        'metrics': {},
        'test_pbar': None,
    }

    # load model
    tester_status = Tester.load_model(cfg, tester_status)
    
    # prepare for progress bar
    tester_status = Tester.get_pbar(cfg, tester_status)

    Tester.before_inference(cfg, tester_status)

    tester_status['metrics'] = test(cfg, tester_status)

    Tester.after_inference(cfg, tester_status)


def infer_portal(infer_cfg):
    setattr(infer_cfg.info, 'infer_start_time', TimeMisc.get_time_str())
    
    # combine train(read) inference(input) configs
    cfg = PortalMisc.combine_train_infer_configs(infer_cfg)
    
    # special config adjustment(debug and work_dir for inference)
    PortalMisc.special_config_adjustment(cfg)
    
    # seed everything
    PortalMisc.seed_everything(cfg)

    # save configs to work_dir as .yaml file (and save current project files if needed)
    PortalMisc.save_configs(cfg)

    # choose whether to print configs of each rank
    PortalMisc.print_config(cfg, force_all_rank=False)

    # init loggers(wandb and local:log_file)
    PortalMisc.init_loggers(cfg)
    
    # interrupt handler
    PortalMisc.interrupt_handler(infer_cfg)

    # main tester
    test_run(cfg)
    
    # end everything
    PortalMisc.end_everything(cfg, end_with_printed_cfg=True)


if __name__ == '__main__':
    infer_cfg = ConfigMisc.get_configs(config_dir='./configs/', default_config_name='train')
    assert hasattr(infer_cfg, 'info'), 'infer_cfg.info not found'

    # init distributed mode
    DistMisc.init_distributed_mode(infer_cfg)
    
    SweepMisc.init_sweep_mode(infer_cfg, infer_portal)
