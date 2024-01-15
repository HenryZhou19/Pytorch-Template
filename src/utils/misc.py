import importlib
import logging
import os
import random
import shutil
import signal
import sys
import time
import warnings
from argparse import Namespace
from collections import UserList
from copy import deepcopy
from glob import glob
from math import inf

import numpy as np
import psutil
import sacred
import torch
import torch.distributed as dist
import yaml
from torch import nn
from tqdm import tqdm
from tqdm.utils import disp_trim

__all__ = [
    'ConfigMisc',
    'PortalMisc',
    'DistMisc',
    'ModelMisc',
    'LoggerMisc',
    'SweepMisc',
    'TensorMisc',
    'ImportMisc',
    'TimeMisc',
    ]

class ConfigMisc:
    @staticmethod
    def get_configs(config_dir):
        main_config_path = ConfigMisc._get_main_config_file_path(config_dir)
        additional_config_paths = ConfigMisc._get_additional_config_file_paths(config_dir, main_config_path)
        return ConfigMisc._get_configs_from_sacred(main_config_path, additional_config_paths)
    
    @staticmethod
    def _get_main_config_file_path(config_dir):
        args = sys.argv
        main_config_name = None
        for arg in args:
            if arg.startswith('config.main='):
                main_config_name = arg.split('=')[1]
                break
        assert main_config_name is not None, 'Should have a main config file name.'
        return os.path.join(config_dir, main_config_name + '.yaml')
    
    @staticmethod
    def _get_additional_config_file_paths(config_dir, main_config_path):
        additional_configs = getattr(ConfigMisc.read_from_yaml(main_config_path).config, 'additional', [])
        for idx, additional_config in enumerate(additional_configs):
            additional_configs[idx] = os.path.join(config_dir, additional_config + '.yaml')
        return additional_configs
    
    @staticmethod
    def _get_configs_from_sacred(main_config_path, additional_config_paths, do_print=False):
        ex = sacred.Experiment('Config Collector', save_git_info=False)
        
        for additional_config_path in additional_config_paths:
            ex.add_config(additional_config_path)
        ex.add_config(main_config_path)
                
        def trim_sacred_configs(_run):
            final_config = _run.config
            final_config.pop('seed', None)  # seed given by sacred is useless
            config_mods = _run.config_modifications
            if do_print:
                if 'RANK' in os.environ:
                    rank = int(os.environ['RANK'])
                    if rank != 0:
                        return vars(config_mods)
                print(f'\nInitial configs read by sacred for ALL Ranks:')
                print(sacred.commands._format_config(final_config, config_mods))
            return vars(config_mods)

        @ex.main
        def get_init_configs(_config, _run):
            modified_configs = trim_sacred_configs(_run)
            return modified_configs

        ex_run = ex.run_commandline()
        cfg = ConfigMisc.nested_dict_to_nested_namespace(ex_run.config)
        cfg.modified_cfg_dict = ex_run.result
        for k, v in cfg.modified_cfg_dict.items():
            cfg.modified_cfg_dict[k] = [cfg_name.split('.')[-1] for cfg_name in v]

        return cfg

    @staticmethod 
    def nested_dict_to_nested_namespace(dictionary, ignore_key_list=[]):
        namespace = dictionary
        if isinstance(dictionary, dict):
            namespace = Namespace(**dictionary)
            for key, value in dictionary.items():
                if key in ignore_key_list:
                    delattr(namespace, key)
                    continue
                setattr(namespace, key, ConfigMisc.nested_dict_to_nested_namespace(value, ignore_key_list))
        return namespace
    
    @staticmethod 
    def nested_namespace_to_nested_dict(namespace, ignore_name_list=[]):
        dictionary = {}
        for name, value in vars(namespace).items():
            if name in ignore_name_list:
                continue
            if isinstance(value, Namespace):
                dictionary[name] = ConfigMisc.nested_namespace_to_nested_dict(value, ignore_name_list)
            else:
                dictionary[name] = value
        return dictionary
    
    @staticmethod
    def nested_namespace_to_plain_namespace(namespace, ignore_name_list=[]):
        def setattr_safely(ns, n, v):
            assert not hasattr(ns, n), f'Namespace conflict: {n}(={v})'
            setattr(ns, n, v)
        
        plain_namespace = Namespace()
        for name, value in vars(namespace).items():
            if name in ignore_name_list:
                continue
            if isinstance(value, Namespace):
                plain_subnamespace = ConfigMisc.nested_namespace_to_plain_namespace(value, ignore_name_list)
                for subname, subvalue in vars(plain_subnamespace).items():
                    setattr_safely(plain_namespace, subname, subvalue)
            else:
                setattr_safely(plain_namespace, name, value)
        
        return plain_namespace
    
    @staticmethod
    def update_nested_namespace(cfg_base, cfg_new):
        for name, value in vars(cfg_new).items():
            if isinstance(value, Namespace):
                if name not in vars(cfg_base):
                    setattr(cfg_base, name, Namespace())
                ConfigMisc.update_nested_namespace(getattr(cfg_base, name), value)
            else:
                setattr(cfg_base, name, value)
                
    @staticmethod
    def setattr_for_nested_namespace(cfg, name_list, value):
        namespace_now = cfg
        for name in name_list[:-1]:
            namespace_now = getattr(namespace_now, name)
        setattr(namespace_now, name_list[-1], value)

    @staticmethod
    def read_from_yaml(path):
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f.read())
        return ConfigMisc.nested_dict_to_nested_namespace(config)

    @staticmethod
    def write_to_yaml(path, config, ignore_name_list):
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(ConfigMisc.nested_namespace_to_nested_dict(config, ignore_name_list), f)

    @staticmethod
    def get_specific_list(cfg, cfg_keys):
        def get_nested_attr(cfg, key):
            if '.' in key:
                key, subkey = key.split('.', 1)
                return get_nested_attr(getattr(cfg, key), subkey)
            else:
                return getattr(cfg, key)
        specific_list = []
        for cfg_key in cfg_keys:
            result = get_nested_attr(cfg, cfg_key)
            if result is not None:
                specific_list.append(str(result))
        return specific_list

    @staticmethod
    def output_dir_extras(cfg):
        extras = '_'.join([cfg.info.start_time] + ConfigMisc.get_specific_list(cfg, cfg.info.name_tags))
        if cfg.special.debug is not None:
            extras = 'debug_' + extras
        return extras
    
    @staticmethod
    def is_inference(cfg):
        return hasattr(cfg, 'tester')


class PortalMisc:
    @staticmethod
    def _find_available_new_path(path, suffix=''):
        if os.path.exists(path):
            counter = 1
            new_path = f'{path}_{suffix}{counter}'
            while os.path.exists(new_path):
                new_path = f'{path}_{suffix}{counter}'
                counter += 1
            return new_path
        else:
            return path
    
    @staticmethod
    def _save_currect_project(cfg):
        if DistMisc.is_main_process():
            main_py_files = glob('./*.py')
            source_paths = [
                './src',
                './scripts',
                './configs',
            ] + main_py_files
            destination_dir = os.path.join(cfg.info.work_dir, 'current_project')
            if cfg.trainer.resume is not None:
                destination_dir += f'_resume_{cfg.info.resume_start_time}.yaml'  
            
            for source_path in source_paths:
                if os.path.isdir(source_path):
                    shutil.copytree(source_path, os.path.join(destination_dir, os.path.basename(source_path)),
                                    ignore=shutil.ignore_patterns('__pycache__'))
                elif os.path.isfile(source_path):
                    shutil.copy(source_path, destination_dir)
                else:
                    print(f'Skipping {source_path} as it is neither a file nor a directory.')
            print(LoggerMisc.block_wrapper('Files and folders of currect project are copied successfully.'))
    
    @staticmethod
    def combine_train_infer_configs(infer_cfg, use_train_seed=True):
        cfg = ConfigMisc.read_from_yaml(infer_cfg.tester.train_cfg_path)  # config in training
        train_seed_base = cfg.seed_base
        ConfigMisc.update_nested_namespace(cfg, infer_cfg)
        if use_train_seed:
            cfg.seed_base = train_seed_base

        cfg.info.train_work_dir = cfg.info.work_dir
        cfg.info.work_dir = cfg.info.train_work_dir + '/inference_results/' + cfg.info.infer_start_time
        cfg.trainer.grad_accumulation = 1
        if DistMisc.is_main_process():
            if not os.path.exists(cfg.info.work_dir):
                os.makedirs(cfg.info.work_dir)
        checkpoint_path = glob(os.path.join(
            cfg.info.train_work_dir,
            'checkpoint_best_epoch_*.pth' if cfg.tester.use_best else 'checkpoint_last_epoch_*.pth'))
        assert len(checkpoint_path) == 1, f'Found {len(checkpoint_path)} checkpoints, please check.'
        cfg.tester.checkpoint_path = checkpoint_path[0]

        return cfg

    @staticmethod 
    def resume_or_new_train_dir(cfg):  # only for train
        assert hasattr(cfg.env, 'distributed')
        if cfg.trainer.resume is not None:  # read 'work_dir', 'start_time' from the .yaml file
            print(LoggerMisc.block_wrapper(f'Resuming from: {cfg.trainer.resume}, reading existing configs...', '>'))
            cfg_old = ConfigMisc.read_from_yaml(cfg.trainer.resume)
            # XXX: assert critial params are the same, but others can be changed(e.g. info...)
            work_dir = cfg_old.info.work_dir
            setattr(cfg.info, 'resume_start_time', cfg.info.start_time)
            cfg.info.start_time = cfg_old.info.start_time
        else:
            work_dir = os.path.join(cfg.info.output_dir, ConfigMisc.output_dir_extras(cfg))
            if DistMisc.is_main_process():
                print(LoggerMisc.block_wrapper(f'New start at: {work_dir}', '>'))
                if not os.path.exists(work_dir):
                    os.makedirs(work_dir)
        cfg.info.work_dir = work_dir

    @staticmethod
    def seed_everything(cfg):
        assert hasattr(cfg.env, 'distributed')
        if cfg.env.seed_with_rank:
            seed_rank = cfg.seed_base + DistMisc.get_rank()
        else:
            seed_rank = cfg.seed_base
        
        os.environ['PYTHONHASHSEED'] = str(seed_rank)

        random.seed(seed_rank)
        np.random.seed(seed_rank)
        
        torch.manual_seed(seed_rank)
        torch.cuda.manual_seed(seed_rank)
        # torch.cuda.manual_seed_all(seed_rank)  # no need here as each process has a different seed

        if cfg.env.cuda_deterministic:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    @staticmethod
    def special_config_adjustment(cfg):
        if cfg.special.debug == 'one_iter':  # 'one_iter' debug mode
            cfg.env.num_workers = 0
        if cfg.trainer.grad_accumulation > 1:
            warnings.warn('Gradient accumulation is set to N > 1. This may affect the function of some modules(e.g. batchnorm, lr_scheduler).')
        cfg.data.batch_size_total = cfg.data.batch_size_per_rank * cfg.env.world_size * cfg.trainer.grad_accumulation
        cfg.info.batch_info = f'{cfg.data.batch_size_total}={cfg.data.batch_size_per_rank}_{cfg.env.world_size}_{cfg.trainer.grad_accumulation}'
        
        if not ConfigMisc.is_inference(cfg):  # only for train
            if cfg.data.sync_lr_with_batch_size > 0:
                cfg.trainer.optimizer.lr_default *= float(cfg.data.batch_size_total) / cfg.data.sync_lr_with_batch_size
                if hasattr(cfg.trainer.optimizer, 'param_groups'):
                    lr_mark = 'lr_'
                    for k, v in vars(cfg.trainer.optimizer.param_groups).items():
                        if k.startswith(lr_mark):
                            setattr(cfg.trainer.optimizer.param_groups, k, v * float(cfg.data.batch_size_total) / cfg.data.sync_lr_with_batch_size)

    @staticmethod
    def save_configs(cfg):
        if DistMisc.is_main_process():
            if not os.path.exists(cfg.info.work_dir):
                os.makedirs(cfg.info.work_dir)
            if cfg.trainer.resume is None:
                cfg_file_name = 'cfg.yaml'
            else:
                cfg_file_name = f'cfg_resume_{cfg.info.resume_start_time}.yaml'  
            
            ConfigMisc.write_to_yaml(os.path.join(cfg.info.work_dir, cfg_file_name), cfg, ignore_name_list=cfg.special.print_save_config_ignore + ['modified_cfg_dict'])
            
        if cfg.special.save_current_project:
            PortalMisc._save_currect_project(cfg)

    @staticmethod
    def print_config(cfg, force_all_rank=False):
        modified_cfg_dict = cfg.modified_cfg_dict
        
        def write_msg_lines(msg_in, cfg_in, indent=1):
            for name in sorted(vars(cfg_in).keys()):
                if name in cfg.special.print_save_config_ignore + ['modified_cfg_dict']:
                    continue   
                m_indent = ' ' * (4 * (indent - 1)) + ' ├─ ' + name
                v = getattr(cfg_in, name)
                if isinstance(v, Namespace):
                    msg_in += write_msg_lines(f'{m_indent}\n', v, indent + 1)
                else:
                    if len(m_indent) > 40:
                        warnings.warn(f'Config key "{name}" with indent is too long (>40) to display, please check.')
                    if len(m_indent) < 38:
                        m_indent += ' ' + '-' * (38 - len(m_indent)) + ' '
                    color_flag = ''
                    if name in modified_cfg_dict['modified']:
                        color_flag = '\033[34m'
                    elif name in modified_cfg_dict['added']:
                        color_flag = '\033[32m'
                    elif name in modified_cfg_dict['typechanged']:
                        color_flag = '\033[31m'
                    elif name in modified_cfg_dict['docs']:
                        color_flag = '\033[30m'
                    msg_in += f'{color_flag}{m_indent:40}{v}\033[0m\n'
            return msg_in

        msg = f'Rank {DistMisc.get_rank()} --- Parameters: (\033[34mmodified, \033[32madded, \033[31mtypechanged, \033[30mdoc)\033[0m\n'
        msg = LoggerMisc.block_wrapper(write_msg_lines(msg, cfg), s='=', block_width=80)

        DistMisc.avoid_print_mess()
        if cfg.env.distributed:
            print(msg, force=force_all_rank)
        else:
            print(msg)
        DistMisc.avoid_print_mess()

    @staticmethod 
    def init_loggers(cfg):
        loggers = Namespace()
        if DistMisc.is_main_process():
            cfg_dict = ConfigMisc.nested_namespace_to_plain_namespace(cfg, cfg.special.logger_config_ignore + ['modified_cfg_dict'])
            if cfg.info.wandb.wandb_enabled:
                import wandb
                wandb_name = '_'.join(ConfigMisc.get_specific_list(cfg, cfg.info.name_tags))
                wandb_name = f'[{cfg.info.task_type}] ' + wandb_name
                wandb_tags = ConfigMisc.get_specific_list(cfg, cfg.info.wandb.wandb_tags)
                if ConfigMisc.is_inference(cfg):
                    wandb_tags.append(f'Infer: {cfg.info.infer_start_time}')
                if cfg.trainer.resume != None:
                    wandb_tags.append(f'Re: {cfg.info.resume_start_time}')
                    if cfg.info.wandb.wandb_resume_enabled:
                        resumed_wandb_id = glob(cfg.info.work_dir + '/wandb/latest-run/*.wandb')[0].split('-')[-1].split('.')[0]
                loggers.wandb_run = wandb.init(
                    project=cfg.info.project_name,
                    name=wandb_name,
                    tags=wandb_tags,
                    dir=cfg.info.work_dir,
                    config=cfg_dict,
                    resume='allow' if cfg.trainer.resume and cfg.info.wandb.wandb_resume_enabled else None,
                    id=resumed_wandb_id if cfg.trainer.resume and cfg.info.wandb.wandb_resume_enabled else None,
                    )
            if cfg.info.tensorboard.tensorboard_enabled:
                from torch.utils.tensorboard import SummaryWriter
                loggers.tensorboard_run = SummaryWriter(log_dir=os.path.join(cfg.info.work_dir, 'tensorboard'))
                # loggers.tensorboard_run.add_hparams(
                #     hparam_dict=ConfigMisc.nested_namespace_to_nested_dict(cfg_dict),
                #     metric_dict={},
                #     )  # tensorboard's hparams logging strategy is not very useful in this case.
            
            if cfg.trainer.resume is None:
                log_file_path = os.path.join(cfg.info.work_dir, 'logs.txt')
            else:
                log_file_path = os.path.join(cfg.info.work_dir, f'logs_resume_{cfg.info.resume_start_time}.txt')       
            loggers.log_file = open(log_file_path, 'a')
            LoggerMisc.print_all_pid(file=loggers.log_file)
        else:
            loggers.log_file = sys.stdout
        return loggers

    @staticmethod 
    def end_everything(cfg, loggers, end_with_printed_cfg=False, force=False):
        if end_with_printed_cfg:
            PortalMisc.print_config(cfg)
        if DistMisc.is_main_process():
            loggers.log_file.close()
            print('log_file closed.')
            try:
                if hasattr(loggers, 'tensorboard_run'):
                    loggers.tensorboard_run.close()
                    print('tensorboard closed.')
                if force:
                    if hasattr(loggers, 'wandb_run'):
                        loggers.wandb_run.finish(exit_code=-1)
                        print('wandb closed.')
                    exit(0)  # 0 for shutting down bash master_port sweeper
                else:
                    if hasattr(loggers, 'wandb_run'):
                        if cfg.special.debug is None:
                            seconds_remain = cfg.info.wandb.wandb_buffer_time - int(TimeMisc.diff_time_str(TimeMisc.get_time_str(), cfg.info.start_time))
                            if seconds_remain > 0:
                                for _ in tqdm(range(seconds_remain), desc='Waiting for wandb to upload all files...'):
                                    time.sleep(1)
                        loggers.wandb_run.finish()
                        print('wandb closed.')
            finally:
                pass
        else:
            if cfg.special.debug is None and cfg.info.wandb.wandb_enabled:
                for _ in range(cfg.info.wandb.wandb_buffer_time):
                    time.sleep(1)

    @staticmethod 
    def interrupt_handler(cfg):
        """Handles SIGINT signal (Ctrl+C) by exiting the program gracefully."""
        def signal_handler(sig, frame):
            if DistMisc.is_main_process():
                print('Caught SIGINT signal, exiting gracefully...')
                LoggerMisc.print_all_pid()
                LoggerMisc.get_wandb_pid(kill_all=True)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)


class DistMisc:
    @staticmethod
    def avoid_print_mess():
        if DistMisc.is_dist_avail_and_initialized():  # 
            dist.barrier()
            time.sleep(DistMisc.get_rank() * 0.1)
    
    @staticmethod
    def all_gather(x: torch.Tensor):

        world_size = DistMisc.get_world_size()
        if world_size == 1:
            return [x]
            
        x_list = [torch.empty_like(x)] * world_size
        dist.all_gather(x_list, x)

        return x_list

    @staticmethod
    def reduce_dict(input_dict, average=True):
        world_size = DistMisc.get_world_size()
        if world_size < 2:
            return input_dict
        with torch.inference_mode():
            # sort the keys so that they are consistent across processes
            names = sorted(input_dict.keys())
            values = torch.stack(list(map(lambda k: input_dict[k]), names), dim=0)
            dist.all_reduce(values)
            if average:
                values /= world_size
            reduced_dict = {k: v for k, v in zip(names, values)}
        return reduced_dict

    @staticmethod
    def reduce(tensor, op='mean'):
        world_size = DistMisc.get_world_size()
        if world_size < 2:
            return tensor
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if op == 'mean':
            tensor = tensor.float() / world_size
        elif op == 'sum':
            pass
        return tensor

    @ staticmethod
    def is_dist_avail_and_initialized():
        return dist.is_available() and dist.is_initialized()

    @staticmethod
    def get_world_size():
        return dist.get_world_size() if DistMisc.is_dist_avail_and_initialized() else 1

    @staticmethod
    def get_rank():
        return dist.get_rank() if DistMisc.is_dist_avail_and_initialized() else 0

    @staticmethod
    def is_main_process():
        return DistMisc.get_rank() == 0

    @staticmethod
    def setup_for_distributed(is_master):
        # This function disables printing when not in master process
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def dist_print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)

        __builtin__.print = dist_print

    @staticmethod
    def init_distributed_mode(cfg):
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            cfg.env.rank = int(os.environ['RANK'])
            cfg.env.world_size = int(os.environ['WORLD_SIZE'])
            cfg.env.gpu = int(os.environ['LOCAL_RANK'])
        elif 'SLURM_PROCID' in os.environ and 'SLURM_PTY_PORT' not in os.environ:
            cfg.env.rank = int(os.environ['SLURM_PROCID'])
            cfg.env.gpu = cfg.env.rank % torch.cuda.device_count()
        else:
            raise NotImplementedError('Must use distributed mode')
            
            # print('Not using distributed mode')
            # cfg.env.distributed = False
            # cfg.env.distributed = 1
            # return
            
        cfg.env.distributed = True
        cfg.env.dist_backend = 'nccl'
        torch.cuda.set_device(cfg.env.gpu)
        
        dist.distributed_c10d.logger.setLevel(logging.WARNING)
        
        dist.init_process_group(
            backend=cfg.env.dist_backend, init_method=cfg.env.dist_url, world_size=cfg.env.world_size, rank=cfg.env.rank
        )       
        # DistMisc.avoid_print_mess()
        # print(f'INFO - distributed init (Rank {cfg.env.rank}): {cfg.env.dist_url}')
        # DistMisc.avoid_print_mess()
        DistMisc.setup_for_distributed(cfg.env.rank == 0)


class ModelMisc:
    @staticmethod
    def print_model_info(cfg, trainer, *args):
        if DistMisc.is_main_process():
            args = set(args)
            if len(args) > 0:
                print_str = ''
                if 'model_structure' in args:
                    print_str += str(trainer.model_without_ddp) + '\n'
                if 'trainable_params' in args:
                    print_str += f'Trainable parameters: {sum(map(lambda p: p.numel() if p.requires_grad else 0, trainer.model_without_ddp.parameters()))}\n'
                if 'total_params' in args:
                    print_str += f'Total parameters: {sum(map(lambda p: p.numel(), trainer.model_without_ddp.parameters()))}\n'
                
                print_str = LoggerMisc.block_wrapper(print_str, s='-', block_width=80)
                print(print_str)
                print(print_str, file=trainer.loggers.log_file)
                trainer.loggers.log_file.flush()
    
    @staticmethod
    def show_model_info(cfg, trainer, torchinfo_columns=None):
        if DistMisc.is_main_process():
            input_data_one_sample = TensorMisc.to(trainer.train_loader.collate_fn([trainer.train_loader.dataset[0]])['inputs'], trainer.device)
            temp_model = deepcopy(trainer.model_without_ddp)
            
            if hasattr(trainer.loggers, 'tensorboard_run'):
                if cfg.info.tensorboard.tensorboard_graph:
                    
                    class WriterWrappedModel(torch.nn.Module):
                        def __init__(self, model):
                            super().__init__()
                            self.model = model
                            
                        def forward(self, inputs):
                            output_tensor_list = []
                            for v in self.model(inputs).values():
                                if isinstance(v, torch.Tensor):
                                    output_tensor_list.append(v)
                            return tuple(output_tensor_list)
                        
                    trainer.loggers.tensorboard_run.add_graph(
                        WriterWrappedModel(temp_model),
                        input_data_one_sample,
                        )
            
            if cfg.info.torchinfo:
                import torchinfo
                torchinfo_columns = torchinfo_columns if torchinfo_columns is not None else [
                    'input_size',
                    'output_size',
                    'num_params',
                    'params_percent',
                    'kernel_size',
                    'mult_adds',
                    'trainable',
                    ]
                assert cfg.data.batch_size_per_rank == trainer.train_loader.batch_size
                
                class TorchinfoWrappedModel(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                        
                    def forward(self, **inputs):
                        return self.model(inputs)
                
                with torch.cuda.amp.autocast(enabled=trainer.scaler is not None):
                    print_str = torchinfo.summary(
                        TorchinfoWrappedModel(temp_model),
                        input_data=TensorMisc.expand_one_sample_to_batch(input_data_one_sample, cfg.data.batch_size_per_rank),
                        col_names=torchinfo_columns,
                        depth=9,
                        verbose=0,
                        )
                # Check model info in OUTPUT_PATH/logs.txt
                print(print_str, file=trainer.loggers.log_file)    
                print(LoggerMisc.block_wrapper(f'torchinfo: Model structure and summary have been saved.'))
            
            if cfg.info.print_param_names:
                print('\nAll Params:', file=trainer.loggers.log_file)
                for k, _ in temp_model.named_parameters():
                    print(f'\t{k}', file=trainer.loggers.log_file)
                print('\n', file=trainer.loggers.log_file)
                
            trainer.loggers.log_file.flush()
            del temp_model, input_data_one_sample
            torch.cuda.empty_cache()
    
    @staticmethod
    def ddp_wrapper(cfg, model_without_ddp):
        if cfg.env.distributed:
            model_without_ddp = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_without_ddp)
            
            return torch.nn.parallel.DistributedDataParallel(model_without_ddp, device_ids=[cfg.env.gpu],
                find_unused_parameters=cfg.env.find_unused_params,
            )
        else:
            return model_without_ddp
        
    @staticmethod
    def load_state_dict_with_more_info(module, state_dict, strict=False, print_keys_level=1):
        missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=strict)
        if print_keys_level > 0:
            missing_keys = list(set(['.'.join(missing_keys.split('.')[:print_keys_level]) for missing_keys in missing_keys]))
            unexpected_keys = list(set(['.'.join(unexpected_key.split('.')[:print_keys_level]) for unexpected_key in unexpected_keys]))
            print_info = 'state_dict loaded not strictly.\n\nMISSING KEYS:\n    ' + '\n    '.join(missing_keys) + '\n\nUNEXPECTED KEYS:\n    ' + '\n    '.join(unexpected_keys)
            print(LoggerMisc.block_wrapper(print_info, '#'))
        
    @staticmethod
    def toggle_batchnorm_track_running_stats(module: nn.Module, true_or_false: bool):
        for child in module.children():
            if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
                child.track_running_stats = true_or_false
            else:
                ModelMisc.toggle_batchnorm_track_running_stats(child, true_or_false)
    
    @staticmethod
    def convert_batchnorm_to_instancenorm(module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm3d):
                setattr(module, name, nn.InstanceNorm3d(child.num_features))
            elif isinstance(child, nn.BatchNorm2d):
                setattr(module, name, nn.InstanceNorm2d(child.num_features))
            elif isinstance(child, nn.BatchNorm1d):
                setattr(module, name, nn.InstanceNorm1d(child.num_features))
            else:
                ModelMisc.convert_batchnorm_to_instancenorm(child)

    @staticmethod
    def update_or_freeze_submodules(module, submodule_name_list, is_trainable: bool, strict=False):  # whether to update the parameters of the submodules
        """
        Just change the trainable property of submodules' parameters.
        
        Some special statistics(e.g. BatchNorm's running mean and variance) are still updated and Dropouts are still working 
        unless the submodules are set to eval mode by calling ModelMisc.train_or_eval_submodules().
        
        """
        for name in submodule_name_list:
            submodule: torch.nn.Module = getattr(module, name, 'None')
            if submodule == 'None':
                if strict:
                    raise ValueError(f'Cannot find submodule "{name}" in {module.__class__.__name__}.')
                else:
                    warnings.warn(f'Cannot find submodule "{name}" in {module.__class__.__name__}.')
            else:
                for param in submodule.parameters():
                    param.requires_grad = is_trainable
    
    @staticmethod
    def train_or_eval_submodules(module, submodule_name_list, is_train: bool, strict=False):
        """
        Just change the behavior of some specific submodules (e.g. BatchNorm, Dropout).
        
        Gradients of the submodules are still computed (and updated if trainable)
        unless the submodules are set to untrainable mode by calling ModelMisc.update_or_freeze_submodules().
        
        """
        for name in submodule_name_list:
            submodule: torch.nn.Module = getattr(module, name, 'None')
            if submodule == 'None':
                if strict:
                    raise ValueError(f'Cannot find submodule "{name}" in {module.__class__.__name__}.')
                else:
                    warnings.warn(f'Cannot find submodule "{name}" in {module.__class__.__name__}.')
            else:
                submodule.train() if is_train else submodule.eval()
        

class LoggerMisc:
    class MultiTQDM:
        def __init__(self, postlines=1, *args, **kwargs) -> None:
            self.bar_main = tqdm(*args, **kwargs)
            self.postlines = postlines
            self.bar_postlines = [tqdm(
                total=0,
                dynamic_ncols=True,
                position=i + 1,
                maxinterval=inf,
                bar_format='{desc}' 
            ) for i in range(postlines)]

        def unpause(self):
            self.bar_main.unpause()

        def update(self, n):
            self.bar_main.update(n)

        def reset(self):
            self.bar_main.reset()

        def close(self):
            self.bar_main.close()
            for bar_postline in self.bar_postlines:
                bar_postline.close()

        def refresh(self):
            self.bar_main.refresh()
            for bar_postline in self.bar_postlines:
                bar_postline.refresh()

        def set_description_str(self, desc, refresh=True):
            self.bar_main.set_description_str(desc, refresh)
        
        def set_postfix_str(self, desc, refresh=True):
            self.bar_main.set_postfix_str(desc, refresh)

        def _trim(self, desc):
            desc = str(desc)
            return disp_trim(desc, self.bar_main.ncols)
        
        def set_postlines_str(self, desc: list, refresh=True):
            assert len(desc) == self.postlines
            for bar_postline, d in zip(self.bar_postlines, desc):
                bar_postline.set_description_str(self._trim(d), refresh)

    @staticmethod
    def block_wrapper(input_object, s='=', block_width=80):
        str_input = str(input_object)
        if not str_input.endswith('\n'):
            str_input += '\n'
        return '\n' + s * block_width + '\n' + str_input + s * block_width + '\n'
    
    @staticmethod
    def logging(loggers, group, output_dict, step):
        if DistMisc.is_main_process():
            if hasattr(loggers, 'wandb_run'):
                for k, v in output_dict.items():
                    if k == 'epoch':
                        loggers.wandb_run.log({k: v}, step=step)  # log epoch without group
                    else:
                        loggers.wandb_run.log({f'{group}/{k}': v}, step=step)
                    # loggers.wandb_run.log({'output_image': [wandb.Image(output_dict['output_image'])]}, step=step)
                    # loggers.wandb_run.log({'output_video': wandb.Video(output_dict['output_video'], fps=30, format='mp4')}, step=step)
            if hasattr(loggers, 'tensorboard_run'):
                for k, v in output_dict.items():
                    if k == 'epoch':
                        loggers.tensorboard_run.add_scalar(k, v, global_step=step)  # log epoch without group
                    else:
                        loggers.tensorboard_run.add_scalar(f'{group}/{k}', v, global_step=step)
                    # loggers.tensorboard_run.add_image("output_image", output_dict['output_image'], global_step=step)
                    # loggers.tensorboard_run.add_image("output_video", output_dict['output_video'], global_step=step)
                    
    @staticmethod
    def print_all_pid(get_parent=True, specific_parent='torchrun', file=sys.stdout):
        p = psutil.Process()
        if get_parent:
            if specific_parent is not None and p.parent().name() != specific_parent:
                return
            p = p.parent()
        p_children = p.children(recursive=True)
        all_processes = '\n'.join([f'    PID: {str(p.pid):9s}Name: {p.name():34s}Parent\'s PID: {p.parent().pid}' for p in [p] + p_children])
        print(LoggerMisc.block_wrapper(f'All sub-processes of {p.name()}:\n{all_processes}', s='#'), file=file)
    
    @staticmethod
    def get_wandb_pid(get_parent=True, specific_parent='torchrun', kill_all=False, kill_wait_time=60):
        p = psutil.Process()
        if get_parent:
            if specific_parent is not None and p.parent().name() != specific_parent:
                return
            p = p.parent()
        p_children = p.children(recursive=True)
        wandb_pid_list = []
        for p in p_children:
            if 'wandb' in p.name():
                wandb_pid_list.append(p.pid)
                if kill_all:
                    os.kill(p.pid, signal.SIGTERM)
                    print(LoggerMisc.block_wrapper(f'wandb process (PID: {p.pid}) may need to be killed manually if it\'s still running.', s='#'))
        return wandb_pid_list


class SweepMisc:
    @staticmethod
    def init_sweep_mode(cfg, portal_fn):
        if cfg.sweep.sweep_enabled:
            if hasattr(cfg, 'trainer'):
                if cfg.trainer.resume is not None:
                    print(LoggerMisc.block_wrapper('Sweep mode cannot be used with resume in phase of training. Ignoring all sweep configs...', '$'))
                    portal_fn(cfg)
            else:
                assert hasattr(cfg, 'tester'), 'Sweep mode can only be used in phase of training or inference.'
            sweep_cfg_dict = ConfigMisc.nested_namespace_to_nested_dict(cfg.sweep.sweep_params)
            
            from itertools import product
            combinations = [dict(zip(sweep_cfg_dict.keys(), values)) for values in product(*sweep_cfg_dict.values())]
            
            for idx, combination in enumerate(combinations):              
                print(LoggerMisc.block_wrapper(f'Sweep mode: [{idx + 1}/{len(combinations)}] combinations', s='#', block_width=80))
                
                cfg_now = deepcopy(cfg)
                for chained_k, v in combination.items():
                    k_list = chained_k.split('//')
                    ConfigMisc.setattr_for_nested_namespace(cfg_now, k_list, v)
                portal_fn(cfg_now)
        else:
            portal_fn(cfg)


class TensorMisc:
    @staticmethod
    def to(data, device, non_blocking=False):
        if isinstance(data, torch.Tensor):
            return data.to(device, non_blocking=non_blocking)
        elif getattr(data, 'not_to_cuda', False) or isinstance(data, (str, int, float)):
            return data
        elif isinstance(data, tuple):
            return tuple(TensorMisc.to(d, device, non_blocking) for d in data)
        elif isinstance(data, list):
            return [TensorMisc.to(d, device, non_blocking) for d in data]
        elif isinstance(data, dict):
            return {k: TensorMisc.to(v, device, non_blocking) for k, v in data.items()}
        else:
            raise TypeError(f'Unknown type: {type(data)}')
        
    class NotToCudaList(UserList):
        not_to_cuda = True
        
    class GradCollector:
        def __init__(self, x: torch.Tensor):
            self.grad = None
            if x.requires_grad:
                x.register_hook(self.hook())  

        def hook(self):
            def _hook(grad):
                self.grad = grad
            return _hook
    
    @staticmethod
    def get_one_sample_from_batch(data, index=0, keep_batch_dim=True):
        if isinstance(data, torch.Tensor) or getattr(data, 'not_to_cuda', False):
            if keep_batch_dim:
                return data[index:index+1]
            else:
                return data[index]
        elif isinstance(data, dict):
            return {k: TensorMisc.get_one_sample_from_batch(v, index, keep_batch_dim) for k, v in data.items()}
        else:
            raise TypeError(f'Unknown type: {type(data)}')
        
    @staticmethod
    def expand_one_sample_to_batch(data, batch_size, have_batch_dim=True):
        assert have_batch_dim, 'Only support expanding one sample to batch when have_batch_dim is True.'
        if isinstance(data, torch.Tensor):
            return data.expand(batch_size, *data.shape[1:])
            # if have_batch_dim:
            #     return data.expand(batch_size, *data.shape[1:])
            # else:
            #     return data.unsqueeze(0).expand(batch_size, *data.shape)
        elif getattr(data, 'not_to_cuda', False):
            return data * batch_size
        elif isinstance(data, dict):
            return {k: TensorMisc.expand_one_sample_to_batch(v, batch_size) for k, v in data.items()}
        else:
            raise TypeError(f'Unknown type: {type(data)}')
        
        
class ImportMisc:
    @staticmethod
    def import_current_dir_all(current_file, current_module_name):
        current_directory = os.path.dirname(current_file)
        current_file_name = os.path.basename(current_file)
        files = os.listdir(current_directory)
        for file in files:
            if file.endswith('.py') and file != current_file_name:
                module_name = os.path.splitext(file)[0]
                importlib.import_module(f'{current_module_name}.{module_name}')      


class TimeMisc:
    @staticmethod
    def get_time_str():
        return time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    
    @staticmethod
    def diff_time_str(time_str_new, time_str_old):
        format_str = '%Y-%m-%d-%H-%M-%S'
        time_new = time.strptime(time_str_new, format_str)
        time_old = time.strptime(time_str_old, format_str)
        return time.mktime(time_new) - time.mktime(time_old)
    
    class Timer:
        def __init__(self):
            self.t_start= time.time()
            self.t = self.t_start
            
        def press(self):
            self.t = time.time()

        def restart(self):
            self.__init__()

        @property
        def info(self):
            now = time.time()
            return {
                'all': now - self.t_start,
                'last': now - self.t
                }
        
    class TimerContext:
        def __init__(self, block_name, print_threshold=0.0, do_print=True):
            self.block_name = block_name
            self.print_threshold = print_threshold
            self.do_print = do_print
            
        def __enter__(self):
            self.timer = TimeMisc.Timer()

        def __exit__(self, *_):
            if self.do_print:
                if self.timer.info['all'] >= self.print_threshold:
                    m_indent = '    ' + self.block_name
                    if len(m_indent) > 40:
                        warnings.warn(f'Block name "{self.block_name}" with indent is too long (>40) to display, please check.')
                    if len(m_indent) < 38:
                            m_indent += ' ' + '-' * (38 - len(m_indent)) + ' '
                    print(f'{m_indent:40s}elapsed time: {self.timer.info["all"]:.4f}')
