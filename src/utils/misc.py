import importlib
# import logging
import os
import random
import shutil
import signal
import sys
import time
import warnings
from collections import UserList, defaultdict
from copy import deepcopy
from glob import glob
from math import inf
from types import SimpleNamespace
from typing import TYPE_CHECKING

from tqdm import tqdm
from tqdm.utils import disp_trim

__all__ = [
    'ImportMisc',
    'ConfigMisc',
    'PortalMisc',
    'DistMisc',
    'ModelMisc',
    'LoggerMisc',
    'SweepMisc',
    'TensorMisc',
    'TimeMisc',
    'DummyContextManager',
    ]


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
    
    class LazyImporter:
        def __init__(self, module_name: str):
            self.module_name = module_name
            self.module = None
        
        def _import(self):
            if self.module is None:
                self.module = importlib.import_module(self.module_name)
            return self.module
        
        def __getattr__(self, name):
            module = self._import()
            return getattr(module, name)


if TYPE_CHECKING:
    import numpy as np
    import psutil
    import torch
    import yaml
    from torch import distributed as dist
    from torch import nn
else:
    np = ImportMisc.LazyImporter('numpy')
    psutil = ImportMisc.LazyImporter('psutil')
    torch = ImportMisc.LazyImporter('torch')
    yaml = ImportMisc.LazyImporter('yaml')
    dist = ImportMisc.LazyImporter('torch.distributed')
    nn = ImportMisc.LazyImporter('torch.nn')


class ConfigMisc:
    @staticmethod
    def get_configs():
        main_config_path = ConfigMisc._get_main_config_file_path()
        additional_config_paths = ConfigMisc._get_additional_config_file_paths(main_config_path)
        cfg = ConfigMisc._parse_yaml_files(additional_config_paths + [main_config_path])
        cfg.modified_cfg_dict = defaultdict(dict)
        cfg = ConfigMisc._update_config_with_cli_args(cfg)
        return cfg
    
    @staticmethod
    def _get_main_config_file_path():
        args = sys.argv
        main_config_path = None
        for arg in args:
            if arg.startswith('config.main='):
                main_config_path = arg.split('=')[1]
                break
        assert main_config_path is not None, 'Should have a main config file name.'
        return main_config_path
    
    @staticmethod
    def _get_additional_config_file_paths(main_config_path):
        additional_config_paths = getattr(ConfigMisc.read_from_yaml(main_config_path).config, 'additional', [])
        return additional_config_paths
    
    @staticmethod
    def _parse_yaml_files(config_file_paths):
        """Load and merge multiple YAML files."""
        cfg = SimpleNamespace()
        for path in config_file_paths:
            cfg_new = ConfigMisc.read_from_yaml(path)
            ConfigMisc.update_nested_namespace(cfg, cfg_new)
        return cfg
    
    @staticmethod
    def _update_config_with_cli_args(cfg):
        args = sys.argv
        for arg in args:
            if "=" not in arg:
                continue  # Skip arguments without "key=value" format
            key_path, value = arg.split("=", 1)
            value = yaml.safe_load(value)  # Convert strings like 'true', '1' properly
            
            keys = key_path.split(".")
            ConfigMisc.setattr_for_nested_namespace(cfg, keys, value, track_modifications=True, mod_dict_key_prefix='cli')
        return cfg
    
    @staticmethod 
    def nested_dict_to_nested_namespace(dictionary, ignore_key_list=[]):
        namespace = dictionary
        if isinstance(dictionary, dict):
            namespace = SimpleNamespace(**dictionary)
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
            if isinstance(value, SimpleNamespace):
                dictionary[name] = ConfigMisc.nested_namespace_to_nested_dict(value, ignore_name_list)
            else:
                dictionary[name] = value
        return dictionary
    
    @staticmethod
    def nested_namespace_to_plain_namespace(namespace, ignore_name_list=[]):
        def setattr_safely(namespace, name, value, identifier=None):
            if identifier is not None:
                name = f'{identifier}|{name}'
            assert not hasattr(namespace, name), f'Namespace conflict: {name}(={value})'
            setattr(namespace, name, value)
        
        plain_namespace = SimpleNamespace()
        identifier = getattr(namespace, 'identifier', None)
        
        for name, value in vars(namespace).items():
            if name in ignore_name_list or name == 'identifier':
                continue
            if isinstance(value, SimpleNamespace):
                plain_subnamespace = ConfigMisc.nested_namespace_to_plain_namespace(value, ignore_name_list)
                for subname, subvalue in vars(plain_subnamespace).items():
                    setattr_safely(plain_namespace, subname, subvalue, identifier)
            else:
                setattr_safely(plain_namespace, name, value, identifier)
        
        return plain_namespace
    
    @staticmethod
    def update_nested_namespace(cfg_base, cfg_new):
        for name, value in vars(cfg_new).items():
            if isinstance(value, SimpleNamespace):
                if name not in vars(cfg_base) or not isinstance(getattr(cfg_base, name), SimpleNamespace):
                    setattr(cfg_base, name, SimpleNamespace())
                ConfigMisc.update_nested_namespace(getattr(cfg_base, name), value)
            else:
                setattr(cfg_base, name, value)
    
    @staticmethod
    def setattr_for_nested_namespace(cfg, name_list, value, track_modifications=False, mod_dict_key_prefix=''):
        namespace_now = cfg
        for name in name_list[:-1]:
            namespace_now = getattr(namespace_now, name, SimpleNamespace())
        if track_modifications:
            modified_cfg_dict = getattr(cfg, 'modified_cfg_dict', defaultdict(dict))
            if mod_dict_key_prefix != '':
                mod_dict_key_prefix += '_'
            if not hasattr(namespace_now, name_list[-1]):
                modified_cfg_dict[f'{mod_dict_key_prefix}added'][name_list[-1]] = {'new_value': value, 'full_key': '.'.join(name_list)}
            else:
                old_value = getattr(namespace_now, name_list[-1])
                if old_value != value:
                    modified_cfg_dict[f'{mod_dict_key_prefix}modified'][name_list[-1]] = {'old_value': old_value, 'new_value': value, 'full_key': '.'.join(name_list)}
                    old_type, new_type = type(old_value), type(value)
                    if old_type != new_type:
                        modified_cfg_dict[f'{mod_dict_key_prefix}typechanged'][name_list[-1]] = {'old_type': old_type.__name__, 'new_type': new_type.__name__, 'full_key': '.'.join(name_list)}
        setattr(namespace_now, name_list[-1], value)
        
    @staticmethod
    def auto_track_setattr(cfg, name_list, value):
        ConfigMisc.setattr_for_nested_namespace(cfg, name_list, value, track_modifications=True, mod_dict_key_prefix='auto')
    
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
    def is_inference(cfg):
        return hasattr(cfg, 'tester')


class PortalMisc:
    @staticmethod
    def set_and_broadcast_start_time(cfg, config_name, string_length=19):
        assert hasattr(cfg, 'info'), 'config field "cfg.info" not found'
        if DistMisc.is_main_process():
            time_string = TimeMisc.get_time_string()
            time_bytes = time_string.encode('utf-8')
            buffer = torch.ByteTensor(list(time_bytes)).to(device=cfg.env.device)
        else:
            buffer = torch.ByteTensor(string_length).to(device=cfg.env.device)
        if DistMisc.is_dist_avail_and_initialized():
            dist.broadcast(buffer, src=0)
            dist.barrier()
        time_string = buffer.cpu().numpy().tobytes().decode('utf-8')
        # print(f"Rank {DistMisc.get_rank()} has time string: {time_string}", force=True)
        ConfigMisc.auto_track_setattr(cfg, ['info', config_name], time_string)
    
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
    def _save_currect_project(cfg, compressed=True):
        if DistMisc.is_main_process():
            main_py_files = glob('./*.py')
            source_paths = [
                './src',
                './scripts',
                './configs',
            ] + main_py_files
            destination_dir = os.path.join(cfg.info.work_dir, 'current_project')
            if cfg.trainer.resume is not None:
                destination_dir += f'_resume_{cfg.info.resume_start_time}'  
            
            for source_path in source_paths:
                if os.path.isdir(source_path):
                    shutil.copytree(source_path, os.path.join(destination_dir, os.path.basename(source_path)),
                                    ignore=shutil.ignore_patterns('__pycache__'))
                elif os.path.isfile(source_path):
                    shutil.copy(source_path, destination_dir)
                else:
                    print(f'Skipping {source_path} as it is neither a file nor a directory.')
                    
            if compressed:
                shutil.make_archive(destination_dir, 'zip', destination_dir)
                shutil.rmtree(destination_dir)
                destination_dir = destination_dir + '.zip'
                print(LoggerMisc.block_wrapper(f'Main project files are successfully copied and compressed to "{destination_dir}"'))
            else:
                print(LoggerMisc.block_wrapper(f'Main project files are successfully copied to "{destination_dir}"'))
    
    @staticmethod
    def combine_train_infer_configs(infer_cfg, use_train_seed=True, custom_work_dir=None):
        ## 1. simply combine the train_cfg and infer_cfg, using infer_cfg to overwrite train_cfg
        cfg = ConfigMisc.read_from_yaml(infer_cfg.tester.train_cfg_path)  # config in training
        train_seed_base = cfg.seed_base
        ConfigMisc.update_nested_namespace(cfg, infer_cfg)
        if use_train_seed:
            ConfigMisc.auto_track_setattr(cfg, ['seed_base'], train_seed_base)
        
        ## 2. confirm the train_work_dir for the "checkpoint_path" and the "(inference_)work_dir"
        ## Note: get it from the train_cfg_path, not in the config itself, as the train_work_dir might have been moved or renamed.
        # ConfigMisc.auto_track_setattr(cfg, ['info', 'train_work_dir'], cfg.info.work_dir)
        ConfigMisc.auto_track_setattr(cfg, ['info', 'train_work_dir'], '/'.join(infer_cfg.tester.train_cfg_path.split('/')[:-1]))
        if os.path.abspath(cfg.info.train_work_dir) != os.path.abspath(cfg.info.work_dir):
            print(LoggerMisc.block_wrapper(f'Folder of "train_cfg_path" in inference_config is different from "work_dir" in train_config.\nThe output folder might have been moved or renamed.', '#'))
        
        ## 3. confirm the checkpoint_path (last or best, or specified) for inference
        if cfg.tester.checkpoint_path is None:
            checkpoint_path = glob(os.path.join(
                cfg.info.train_work_dir,
                'checkpoint_best_epoch_*.pth' if cfg.tester.use_best else 'checkpoint_last_epoch_*.pth'))
            assert len(checkpoint_path) == 1, f'Found {len(checkpoint_path)} checkpoints, please check.'
            ConfigMisc.auto_track_setattr(cfg, ['tester', 'checkpoint_path'], checkpoint_path[0])
        else:
            assert os.path.exists(cfg.tester.checkpoint_path), f'Checkpoint path "{cfg.tester.checkpoint_path}" not found.'
        
        ## 4. set work_dir for inference (default: train_work_dir/inference_results)
        if custom_work_dir is not None:
            ConfigMisc.auto_track_setattr(cfg, ['info', 'work_dir'],
                                          os.path.join(custom_work_dir, LoggerMisc.output_dir_time_and_extras(cfg, is_infer=True)))
        else:
            ConfigMisc.auto_track_setattr(cfg, ['info', 'work_dir'],
                                          cfg.info.train_work_dir + '/inference_results/' + LoggerMisc.output_dir_time_and_extras(cfg, is_infer=True))
        if DistMisc.is_main_process():
            if not os.path.exists(cfg.info.work_dir):
                os.makedirs(cfg.info.work_dir)
        
        return cfg
    
    @staticmethod 
    def resume_or_new_train_dir(cfg):  # only for train
        assert hasattr(cfg.env, 'distributed')
        if cfg.trainer.resume is not None:  # read 'work_dir', 'start_time' from the .yaml file
            print(LoggerMisc.block_wrapper(f'Resuming from: {cfg.trainer.resume}, reading existing configs...', '>'))
            cfg_old = ConfigMisc.read_from_yaml(cfg.trainer.resume)
            # XXX: assert critial params are the same, but others can be changed(e.g. info...)
            work_dir = cfg_old.info.work_dir
            ConfigMisc.auto_track_setattr(cfg, ['info', 'resume_start_time'], cfg.info.start_time)
            ConfigMisc.auto_track_setattr(cfg, ['info', 'start_time'], cfg_old.info.start_time)
        else:
            work_dir = os.path.join(cfg.info.output_dir, LoggerMisc.output_dir_time_and_extras(cfg))
            if DistMisc.is_main_process():
                print(LoggerMisc.block_wrapper(f'New start at: {work_dir}', '>'))
                if not os.path.exists(work_dir):
                    os.makedirs(work_dir)
        ConfigMisc.auto_track_setattr(cfg, ['info', 'work_dir'], work_dir)
    
    @staticmethod
    def seed_everything(cfg):
        if cfg.env.seed_with_rank:
            seed_rank = cfg.seed_base + DistMisc.get_rank()
        else:
            seed_rank = cfg.seed_base
        
        os.environ['PYTHONHASHSEED'] = str(seed_rank)
        
        random.seed(seed_rank)
        np.random.seed(seed_rank)
        
        torch.manual_seed(seed_rank)
        if cfg.env.device == 'cuda':
            torch.cuda.manual_seed(seed_rank)
            # torch.cuda.manual_seed_all(seed_rank)  # no need here as each process has a different seed
            
            if cfg.env.cuda_deterministic:
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    
    @staticmethod
    def special_config_adjustment(cfg):
        def _set_real_batch_size_and_lr(cfg):
            if not ConfigMisc.is_inference(cfg):  # for train and val
                if cfg.trainer.grad_accumulation > 1:
                    warnings.warn('Gradient accumulation is set to N > 1. This may affect the function of some modules(e.g. batchnorm, lr_scheduler).')

                ConfigMisc.auto_track_setattr(cfg, ['trainer', 'trainer_batch_size_total'],
                                              cfg.trainer.trainer_batch_size_per_rank * cfg.env.world_size * cfg.trainer.grad_accumulation)
                ConfigMisc.auto_track_setattr(cfg, ['info', 'batch_info'],
                                              f'{cfg.trainer.trainer_batch_size_total}={cfg.trainer.trainer_batch_size_per_rank}_{cfg.env.world_size}_{cfg.trainer.grad_accumulation}')
                
                ConfigMisc.auto_track_setattr(cfg, ['trainer', 'name_optimizers'],
                                              [attr for attr in dir(cfg.trainer) if attr.startswith('optimizer')])
                if cfg.trainer.sync_lr_with_batch_size > 0:
                    for name_optimizer in cfg.trainer.name_optimizers:
                        optimizer_cfg = getattr(cfg.trainer, name_optimizer)
                        ConfigMisc.auto_track_setattr(cfg, ['trainer', name_optimizer, 'lr_default'],
                                                    optimizer_cfg.lr_default * float(cfg.trainer.trainer_batch_size_total) / cfg.trainer.sync_lr_with_batch_size)
                        if hasattr(optimizer_cfg, 'param_groups'):
                            lr_mark = 'lr_'
                            for k, v in vars(optimizer_cfg.param_groups).items():
                                if k.startswith(lr_mark):
                                    ConfigMisc.auto_track_setattr(cfg, ['trainer', 'optimizer', 'param_groups', k],
                                                                v * float(cfg.trainer.trainer_batch_size_total) / cfg.trainer.sync_lr_with_batch_size)
            else: # for inference
                ConfigMisc.auto_track_setattr(cfg, ['tester', 'tester_batch_size_total'],
                                              cfg.tester.tester_batch_size_per_rank * cfg.env.world_size)
                ConfigMisc.auto_track_setattr(cfg, ['info', 'batch_info'],
                                              f'{cfg.tester.tester_batch_size_total}={cfg.tester.tester_batch_size_per_rank}_{cfg.env.world_size}')
        
        _set_real_batch_size_and_lr(cfg)
        
        if cfg.special.debug == 'one_iter':  # 'one_iter' debug mode
            ConfigMisc.auto_track_setattr(cfg, ['env', 'num_workers'], 0)
         
        if cfg.special.no_logger:
            ConfigMisc.auto_track_setattr(cfg, ['info', 'wandb', 'wandb_enabled'], False)
            ConfigMisc.auto_track_setattr(cfg, ['info', 'tensorboard', 'tensorboard_enabled'], False)
            
        if cfg.special.single_eval:
            ConfigMisc.auto_track_setattr(cfg, ['trainer', 'dist_eval'], False)


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
            PortalMisc._save_currect_project(cfg, compressed=True)
            
        if cfg.special.print_config_start:
            PortalMisc._print_config(cfg, force_all_rank=cfg.special.print_config_all_rank)
        else:
            PortalMisc._print_config(cfg, force_all_rank=False, modified_config_only=True)
    
    @staticmethod
    def _print_config(cfg, force_all_rank=False, modified_config_only=False):
        UNCHANGED = 0
        ADDED = 1
        MODIFIED = 2
        TYPECHANGED = 3
        
        FADED_COLOR = '\033[30m'  # black
        COLORS = {
            0: '\033[0m',  # white
            1: '\033[32m',  # green
            2: '\033[34m',  # blue
            3: '\033[31m',  # red
        }
        CLI_COLOR = '\033[33m'  # yellow
        SWEEP_COLOR = '\033[35m'  # magenta
        AUTO_COLOR = '\033[36m'  # cyan
        
        modified_cfg_dict = cfg.modified_cfg_dict
        dict_key_prefix_list = ['auto', 'sweep', 'cli']  # In reverse order of occurrence
        
        def get_value_prefix(dict_key_prefix):
            color = CLI_COLOR if dict_key_prefix == 'cli' else SWEEP_COLOR if dict_key_prefix == 'sweep' else AUTO_COLOR
            value_prefix = color + dict_key_prefix + ': ' + COLORS[UNCHANGED]
            return value_prefix
            
        def check_modified_cfg_dict(modified_cfg_dict, key, value, dict_key_prefix='cli'):
            modified = UNCHANGED
            str_pre, str_post = '', ''
            assert dict_key_prefix in ['cli', 'sweep', 'auto'], f'Invalid dict_key_prefix: {dict_key_prefix}'
            
            if key in modified_cfg_dict[f'{dict_key_prefix}_added']:
                modified = ADDED
                str_post = f'{get_value_prefix(dict_key_prefix)}{COLORS[ADDED]}{value}{COLORS[UNCHANGED]}'  # green
            elif key in modified_cfg_dict[f'{dict_key_prefix}_modified']:
                old_value = modified_cfg_dict[f"{dict_key_prefix}_modified"][key]["old_value"]
                new_value = modified_cfg_dict[f'{dict_key_prefix}_modified'][key]['new_value']
                if key in modified_cfg_dict[f'{dict_key_prefix}_typechanged']:
                    modified = TYPECHANGED
                    old_type = modified_cfg_dict[f"{dict_key_prefix}_typechanged"][key]["old_type"]
                    new_type = modified_cfg_dict[f"{dict_key_prefix}_typechanged"][key]["new_type"]
                    str_pre = f'{FADED_COLOR}{old_value} ({old_type})'
                    str_post = f'{FADED_COLOR} -> {get_value_prefix(dict_key_prefix)}{COLORS[TYPECHANGED]}{new_value} ({new_type}){COLORS[UNCHANGED]}'  # red
                else:
                    modified = MODIFIED
                    str_pre = f'{FADED_COLOR}{old_value}'
                    str_post = f'{FADED_COLOR} -> {get_value_prefix(dict_key_prefix)}{COLORS[MODIFIED]}{new_value}{COLORS[UNCHANGED]}'  # blue
                    
            return modified, str_pre, str_post
        
        def write_config_lines(str_block_in, cfg_in, indent=0):
            str_block_add = ''
            for key in sorted(vars(cfg_in).keys()):
                if key in cfg.special.print_save_config_ignore + ['modified_cfg_dict']:
                    continue   
                key_indent = ' ' * (4 * indent) + ' ├─ ' + key
                value = getattr(cfg_in, key)
                if isinstance(value, SimpleNamespace):
                    str_block_add += write_config_lines(f'{key_indent}\n', value, indent + 1)
                else:
                    if len(key_indent) > 40:
                        warnings.warn(f'Config key "{key}" with indent is too long (>40) to display, please check.')
                    elif len(key_indent) < 38:
                        key_indent += ' ' + '-' * (38 - len(key_indent)) + ' '
                    
                    ever_modified = False
                    str_add, final_str_pre = '', ''
                    for dict_key_prefix in dict_key_prefix_list:
                        modified, str_pre, str_post = check_modified_cfg_dict(modified_cfg_dict, key, value, dict_key_prefix=dict_key_prefix)         
                        if modified:
                            ever_modified = True
                            final_str_pre = f'{COLORS[modified]}{key_indent:40}{COLORS[UNCHANGED]}' + str_pre
                            str_add = str_post + str_add
                            
                    if not modified_config_only and not ever_modified:
                        str_add = f'{key_indent:40}{value}\n'
                    else:
                        str_add = final_str_pre + str_add
                        if str_add != '':
                            str_add += '\n'
                        
                    str_block_add += str_add
            if str_block_add != '':
                return str_block_in + str_block_add
            else:
                return ''
        
        str_block = f'Rank {DistMisc.get_rank()} --- {"Modified" if modified_config_only else "All"} Parameters: (\033[32madded, \033[34mmodified, \033[31mtypechanged)\033[0m\n'
        str_block = LoggerMisc.block_wrapper(write_config_lines(str_block, cfg), s='=', block_width=80)
        
        DistMisc.avoid_print_mess()
        print(str_block, force=force_all_rank)
        DistMisc.avoid_print_mess()
    
    @staticmethod 
    def init_loggers(cfg):
        loggers = SimpleNamespace()
        if DistMisc.is_main_process():
            cfg_plain = ConfigMisc.nested_namespace_to_plain_namespace(cfg, cfg.special.logger_config_ignore + ['modified_cfg_dict'])
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
                    config=cfg_plain,
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
                log_file_path = os.path.join(cfg.info.work_dir, 'logs.log')
            else:
                log_file_path = os.path.join(cfg.info.work_dir, f'logs_resume_{cfg.info.resume_start_time}.log')       
            loggers.log_file = open(log_file_path, 'a')
            if cfg.env.distributed:
                LoggerMisc.print_all_pid(file=loggers.log_file)
            else:
                LoggerMisc.print_all_pid(get_parent=False, file=loggers.log_file)
            loggers.log_file.flush()
        else:
            loggers.log_file = sys.stdout
        return loggers
    
    @staticmethod 
    def end_everything(cfg, loggers, force=False):
        if cfg.special.print_config_end:
            PortalMisc._print_config(cfg, force_all_rank=cfg.special.print_config_all_rank)
        seconds_remain = cfg.info.wandb.wandb_buffer_time - int(TimeMisc.diff_time_str(TimeMisc.get_time_string(), cfg.info.start_time))
        if DistMisc.is_main_process():
            loggers.log_file.close()
            print('log_file closed.')
            try:
                if hasattr(loggers, 'tensorboard_run'):
                    loggers.tensorboard_run.close()
                    print('tensorboard closed.')
                if force:
                    DistMisc.destroy_process_group()
                    
                    if hasattr(loggers, 'wandb_run'):
                        loggers.wandb_run.finish(exit_code=-1)
                        print('wandb closed.')
                    exit(0)  # 0 for shutting down bash master_port sweeper
                else:
                    if hasattr(loggers, 'wandb_run'):
                        if cfg.special.debug is None:
                            if seconds_remain > 0:
                                for _ in tqdm(range(seconds_remain), desc='Waiting for wandb to upload all files...'):
                                    time.sleep(1)
                        loggers.wandb_run.finish()
                        print('wandb closed.')
            finally:
                pass
        else:
            if cfg.special.debug is None and cfg.info.wandb.wandb_enabled:
                if seconds_remain > 0:
                    for _ in range(seconds_remain):
                        time.sleep(1)
    
    @staticmethod 
    def interrupt_handler(cfg):
        """Handles SIGINT signal (Ctrl+C) by exiting the program gracefully."""
        def signal_handler(sig, frame):
            if DistMisc.is_main_process():
                print('Caught SIGINT signal, exiting gracefully...')
                if cfg.env.distributed:
                    LoggerMisc.print_all_pid()
                    LoggerMisc.get_wandb_pid(kill_all=True)
                else:
                    LoggerMisc.print_all_pid(get_parent=False)
                    LoggerMisc.get_wandb_pid(get_parent=False, kill_all=True)
            raise KeyboardInterrupt
        
        signal.signal(signal.SIGINT, signal_handler)


class DistMisc:
    @staticmethod
    def barrier():
        if DistMisc.is_dist_avail_and_initialized():
            dist.barrier()    
    
    @staticmethod
    def avoid_print_mess(sleep_interval=0.1):
        if DistMisc.is_dist_avail_and_initialized():  # 
            dist.barrier()
            time.sleep(DistMisc.get_rank() * sleep_interval)
    
    @staticmethod
    def all_gather(x, concat_out=False):
        '''
        x: [N, *]
        N can be different on different processes
        Make sure (*) is the same shape on all processes
        '''
        x: torch.Tensor
        world_size = DistMisc.get_world_size()
        if world_size == 1:
            x_list = [x]
        else:
            N = torch.tensor(x.shape[0], dtype=torch.int, device=x.device)
            N_list = [torch.zeros(1, dtype=torch.int, device=x.device) for _ in range(world_size)]
            dist.all_gather(N_list, N)
                
            x_list = [torch.empty(N.item(), *x.shape[1:], dtype=x.dtype, device=x.device) for N in N_list]
            dist.all_gather(x_list, x)
        
        if concat_out:
            return torch.cat(x_list, dim=0)
        else:
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
    def is_node_first_rank():
        return int(os.environ['LOCAL_RANK']) == 0
    
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
        if cfg.env.device == 'cuda':
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:  # 
                ConfigMisc.auto_track_setattr(cfg, ['env', 'world_size'], int(os.environ['WORLD_SIZE']))
                ConfigMisc.auto_track_setattr(cfg, ['env', 'rank'], int(os.environ['RANK']))
                ConfigMisc.auto_track_setattr(cfg, ['env', 'local_rank'], int(os.environ['LOCAL_RANK']))
            # elif 'SLURM_PROCID' in os.environ and 'SLURM_PTY_PORT' not in os.environ:
            #     ConfigMisc.auto_track_setattr(cfg, ['env', 'rank'], int(os.environ['SLURM_PROCID']))
            #     ConfigMisc.auto_track_setattr(cfg, ['env', 'local_rank'], cfg.env.rank % torch.cuda.device_count())
                
                ConfigMisc.auto_track_setattr(cfg, ['env', 'distributed'], True)
                ConfigMisc.auto_track_setattr(cfg, ['env', 'dist_backend'], 'nccl')
                ConfigMisc.auto_track_setattr(cfg, ['env', 'dist_url'], 'env://')
                torch.cuda.set_device(cfg.env.local_rank)
            
                # dist.distributed_c10d.logger.setLevel(logging.WARNING)  # this line may cause the multi-machine ddp to hang
                
                dist.init_process_group(
                    backend=cfg.env.dist_backend, init_method=cfg.env.dist_url, world_size=cfg.env.world_size, rank=cfg.env.rank
                )
                # DistMisc.avoid_print_mess()
                # print(f'INFO - distributed init (Rank {cfg.env.rank}): {cfg.env.dist_url}')
                # DistMisc.avoid_print_mess()
            else:
                ConfigMisc.auto_track_setattr(cfg, ['env', 'distributed'], False)
                ConfigMisc.auto_track_setattr(cfg, ['env', 'world_size'], 1)
                ConfigMisc.auto_track_setattr(cfg, ['env', 'rank'], 0)
                ConfigMisc.auto_track_setattr(cfg, ['env', 'local_rank'], 0)
                ConfigMisc.auto_track_setattr(cfg, ['env', 'dist_backend'], 'None')
                ConfigMisc.auto_track_setattr(cfg, ['env', 'dist_url'], 'None')

            DistMisc.setup_for_distributed(cfg.env.rank == 0)
            
            if cfg.env.rank != cfg.env.local_rank:
                if DistMisc.is_node_first_rank():
                    print(LoggerMisc.block_wrapper(f'This is not the main node, which is on {os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}'), force=True)
        elif cfg.env.device == 'cpu':
            ConfigMisc.auto_track_setattr(cfg, ['env', 'distributed'], False)
            ConfigMisc.auto_track_setattr(cfg, ['env', 'world_size'], 1)
            ConfigMisc.auto_track_setattr(cfg, ['env', 'rank'], 0)
            ConfigMisc.auto_track_setattr(cfg, ['env', 'local_rank'], 0)
            ConfigMisc.auto_track_setattr(cfg, ['env', 'dist_backend'], 'None')
            ConfigMisc.auto_track_setattr(cfg, ['env', 'dist_url'], 'None')
            
            DistMisc.setup_for_distributed(True)
            
            if getattr(cfg.amp, 'amp_enabled', False):  # in train mode, check AMP
                print(LoggerMisc.block_wrapper('AMP is not supported on CPU. Automatically turning off AMP by setting "cfg.amp.amp_enabled = False".', '#'))
                ConfigMisc.auto_track_setattr(cfg, ['amp', 'amp_enabled'], False)
            if cfg.env.pin_memory:
                print(LoggerMisc.block_wrapper('Pin memory is not supported on CPU. Automatically turning off pin_memory by setting "cfg.env.pin_memory = False".', '#'))
                ConfigMisc.auto_track_setattr(cfg, ['env', 'pin_memory'], False)
        else:
            raise ValueError('Invalid device type.')
    
    @staticmethod
    def destroy_process_group():
        if dist.is_initialized():
            dist.destroy_process_group()


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
            temp_model = trainer.model_without_ddp
            temp_model.eval()
            
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
                assert cfg.trainer.trainer_batch_size_per_rank == trainer.train_loader.batch_size
                
                class TorchinfoWrappedModel(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                        
                    def forward(self, **inputs):
                        return self.model(inputs)
                
                with trainer.train_autocast():
                    print_str = torchinfo.summary(
                        TorchinfoWrappedModel(temp_model),
                        input_data=TensorMisc.expand_one_sample_to_batch(input_data_one_sample, cfg.trainer.trainer_batch_size_per_rank),
                        col_names=torchinfo_columns,
                        depth=9,
                        verbose=0,
                        )
                # Check model info in OUTPUT_PATH/logs.log
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
            
            return torch.nn.parallel.DistributedDataParallel(model_without_ddp, device_ids=[cfg.env.local_rank],
                find_unused_parameters=cfg.env.find_unused_parameters,
            )
        else:
            return model_without_ddp
    
    @staticmethod
    def load_state_dict_with_more_info(module, state_dict, strict=False, print_keys_level=1):
        missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=strict)
        if print_keys_level > 0:
            matched_keys = list(set(state_dict.keys()) - set(missing_keys) - set(unexpected_keys))
            
            matched_keys = list(set(['.'.join(matched_key.split('.')[:print_keys_level]) for matched_key in matched_keys]))
            missing_keys = list(set(['.'.join(missing_keys.split('.')[:print_keys_level]) for missing_keys in missing_keys]))
            unexpected_keys = list(set(['.'.join(unexpected_key.split('.')[:print_keys_level]) for unexpected_key in unexpected_keys]))
            print_info = f'state_dict loaded not strictly.' \
                + '\n\033[32m\nMATCHED KEYS:\n\033[0m    ' + '\n    '.join(matched_keys) \
                + '\n\033[33m\nMISSING KEYS (only in model):\n\033[0m    ' + '\n    '.join(missing_keys) \
                + '\n\033[34m\nUNEXPECTED KEYS (only in pth):\n\033[0m    ' + '\n    '.join(unexpected_keys)
            print(LoggerMisc.block_wrapper(print_info, '#'))
    
    @staticmethod
    def toggle_batchnorm_track_running_stats(module, true_or_false: bool):
        module: nn.Module
        for child in module.children():
            if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
                child.track_running_stats = true_or_false
            else:
                ModelMisc.toggle_batchnorm_track_running_stats(child, true_or_false)
    
    @staticmethod
    def convert_batchnorm_to_instancenorm(module):
        module: nn.Module
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
    def unfreeze_or_freeze_submodules(module, submodule_name_list, is_trainable: bool, strict=False, verbose=True):  # whether to update the parameters of the submodules
        """
        Just change the trainable property of submodules' parameters.
        
        Some special statistics(e.g. BatchNorm's running mean and variance) are still updated and Dropouts are still working 
        unless the submodules are set to eval mode by calling ModelMisc.train_or_eval_submodules().
        
        verbose: (default True) as this function is usually called only once --- before all epochs
        """
        module: nn.Module
        named_modules = dict(module.named_modules())
        verbose_string = f'{"Unfreeze" if is_trainable else "Freeze"} parameters of the following submodules:'
        for submodule_name in submodule_name_list:
            submodule: torch.nn.Module = named_modules.get(submodule_name, None)
            if submodule is None:
                error_message = f'Cannot find submodule "{submodule_name}" in {module.__class__.__name__} when {"unfreezing" if is_trainable else "freezing"} parameters.'
                if strict:
                    raise ValueError(error_message)
                else:
                    warnings.warn(error_message)
            else:
                verbose_string += f'\n    {submodule_name}'
                for param in submodule.parameters():
                    param.requires_grad = is_trainable
        if verbose and len(submodule_name_list) > 0:
            print(LoggerMisc.block_wrapper(verbose_string, '='))
    
    @staticmethod
    def unfreeze_or_freeze_params(module, params_name_list, is_trainable: bool, strict=False, verbose=True):  # whether to update the parameters of the submodules
        """
        Just change the trainable property of specific parameters.
        
        verbose: (default True) as this function is usually called only once --- before all epochs
        """
        module: nn.Module
        verbose_string = f'{"Unfreeze" if is_trainable else "Freeze"} the following specific parameters:'
        if len(params_name_list) > 0:
            params_dict = dict(module.named_parameters())
            for param_name in params_name_list:
                param = params_dict.get(param_name, None)
                if param is None:
                    error_message = f'Cannot find parameter "{param_name}" in {module.__class__.__name__} when {"unfreezing" if is_trainable else "freezing"} parameters.'
                    if strict:
                        raise ValueError(error_message)
                    else:
                        warnings.warn(error_message)
                else:
                    verbose_string += f'\n    {param_name}'
                    param.requires_grad = is_trainable

        if verbose and len(params_name_list) > 0:
            print(LoggerMisc.block_wrapper(verbose_string, '='))        
    
    @staticmethod
    def train_or_eval_submodules(module, submodule_name_list, is_train: bool, strict=False, verbose=False):
        """
        Just change the behavior of some specific submodules (e.g. BatchNorm, Dropout).
        
        Gradients of the submodules are still computed (and updated if trainable)
        unless the submodules are set to untrainable mode by calling ModelMisc.unfreeze_or_freeze_submodules().
        
        verbose: (default False) as this function is usually called multiple times --- before one (each) epoch
        """
        module: nn.Module
        named_modules = dict(module.named_modules())
        verbose_string = f'Set nn.Module mode of the following submodules to {"train" if is_train else "eval"}:'
        for submodule_name in submodule_name_list:
            submodule: torch.nn.Module = named_modules.get(submodule_name, None)
            if submodule is None:
                error_message = f'Cannot find submodule "{submodule_name}" in {module.__class__.__name__} when setting nn.Module mode to {"train" if is_train else "eval"}'
                if strict:
                    raise ValueError(error_message)
                else:
                    warnings.warn(error_message)
            else:
                verbose_string += f'\n    {submodule_name}'
                submodule.train() if is_train else submodule.eval()
        if verbose and len(submodule_name_list) > 0:
            print(LoggerMisc.block_wrapper(verbose_string, '='))
    
    @staticmethod
    def _re_init_check(module, param_name):
        module: nn.Module
        if hasattr(getattr(module, param_name, None), '_no_reinit'):
            print(f'No re-init for {module}\'s {param_name}')
            return False
        return True


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
    def set_dict_key_prefix(input_dict: dict, prefix):
        return {
            f'{prefix}{k}': v for k, v in input_dict.items()
            }
    
    @staticmethod
    def list_to_multiline_string(items: list, prefix='\t', suffix=''):
        return '\n'.join([prefix + str(item) + suffix for item in items])
    
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
                    # loggers.tensorboard_run.add_image('output_image', output_dict['output_image'], global_step=step)
                    # loggers.tensorboard_run.add_image('output_video', output_dict['output_video'], global_step=step)
    
    @staticmethod
    def print_all_pid(get_parent=True, specific_parent=['torchrun', 'pt_main_thread'], file=sys.stdout):
        p = psutil.Process()
        if get_parent:
            if specific_parent is not None and p.parent().name() not in specific_parent:
                return
            p = p.parent()
        p_children = p.children(recursive=True)
        all_processes = '\n'.join([f'    PID: {str(p.pid):9s}Name: {p.name():34s}Parent\'s PID: {p.parent().pid}' for p in [p] + p_children])
        print(LoggerMisc.block_wrapper(f'All sub-processes of {p.name()}:\n{all_processes}', s='#'), file=file)
    
    @staticmethod
    def get_wandb_pid(get_parent=True, specific_parent=['torchrun', 'pt_main_thread'], kill_all=False, kill_wait_time=60):
        p = psutil.Process()
        if get_parent:
            if specific_parent is not None and p.parent().name() not in specific_parent:
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
    
    @staticmethod
    def output_dir_time_and_extras(cfg, is_infer=False):
        extras = '_'.join([cfg.info.infer_start_time if is_infer else cfg.info.start_time] + ConfigMisc.get_specific_list(cfg, cfg.info.name_tags))
        if cfg.special.debug is not None:
            extras = 'debug_' + extras
        return extras


class SweepMisc:
    @staticmethod
    def _send_email(cfg, subject, message='No-reply'):
        if DistMisc.is_main_process():
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText
            
            email_host = cfg.special.email_host
            email_sender = cfg.special.email_sender
            email_password = cfg.special.email_password
            email_receiver = cfg.special.email_receiver
            try:
                import socket
                device_name = socket.gethostname()
            except:
                device_name = 'Unknown Device'

            msg = MIMEMultipart()
            msg['From'] = device_name
            msg['To'] = 'Base'
            msg['Subject'] = f'{device_name}: {subject}'
            mail_msg = '''<p>{}</p>'''.format(message)
            msg.attach(MIMEText(mail_msg, 'html', 'utf-8'))

            smtp = smtplib.SMTP()
            smtp.connect(email_host, 25)
            smtp.login(email_sender, email_password)
            smtp.sendmail(email_sender, email_receiver, msg.as_string())
            smtp.quit()
            print(LoggerMisc.block_wrapper(f'Email (subject: {subject}) sent successfully.'))
    
    @staticmethod
    def _do_sweep(cfg, portal_fn):
        if hasattr(cfg, 'trainer'):
            if cfg.trainer.resume is not None:
                print(LoggerMisc.block_wrapper('Sweep mode cannot be used with resume in phase of training. Ignoring all sweep configs...', '$'))
                portal_fn(cfg)
                exit()
        else:
            assert hasattr(cfg, 'tester'), 'Sweep mode can only be used in phase of training or inference.'
        sweep_cfg_dict = ConfigMisc.nested_namespace_to_nested_dict(cfg.sweep.sweep_params)
        
        from itertools import product
        combinations = [dict(zip(sweep_cfg_dict.keys(), values)) for values in product(*sweep_cfg_dict.values())]
        
        sweep_skip_indices = getattr(cfg.sweep,'sweep_skip_indices', [])
        sweep_skip_indices = set([x for x in sweep_skip_indices if isinstance(x, int) and x < len(combinations)])
        filtered_combinations = [combination for idx, combination in enumerate(combinations) if idx not in sweep_skip_indices]
        
        for idx, combination in enumerate(filtered_combinations):
            print(LoggerMisc.block_wrapper(f'Sweep mode: [{idx + 1}/{len(filtered_combinations)}] combinations', s='#', block_width=80))
            
            cfg_now = deepcopy(cfg)
            for chained_k, v in combination.items():
                k_list = chained_k.split('//')
                ConfigMisc.setattr_for_nested_namespace(cfg_now, k_list, v, track_modifications=True, mod_dict_key_prefix='sweep')
            portal_fn(cfg_now)
    
    @staticmethod
    def init_sweep_mode(cfg, portal_fn):
        if_send_email = getattr(cfg.special, 'send_email', False)
        email_subject='Elusive Error'
        email_message='Unexpected exit.'
        
        try:
            if cfg.sweep.sweep_enabled:
                SweepMisc._do_sweep(cfg, portal_fn)
            else:
                portal_fn(cfg)
        
        except Exception as e:
            email_subject = 'Error'
            email_message = str(e)
            raise e
        except KeyboardInterrupt:
            email_subject = 'Interrupted'
            email_message = 'KeyboardInterrupt.'
            if not getattr(cfg.special, 'email_when_interrupted', False):
                if_send_email = False
        else:
            email_subject = 'Success'
            email_message = 'Finished.'
        finally:
            if if_send_email:
                SweepMisc._send_email(cfg, email_subject, email_message)   


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
        def __init__(self, x):
            x: torch.Tensor
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
    
    @staticmethod
    def get_gpu_memory_usage(verbose=False, device='cuda'):
        allocated_bytes = torch.cuda.memory_allocated(device=device)
        max_allocated_bytes = torch.cuda.max_memory_allocated(device=device)
        reserved_bytes = torch.cuda.memory_reserved(device=device)
        total_bytes = torch.cuda.get_device_properties(device=device).total_memory
        
        allocated_mb = allocated_bytes / 1048576
        max_allocated_mb = max_allocated_bytes / 1048576
        reserved_mb = reserved_bytes / 1048576
        total_mb = total_bytes / 1048576
        
        if verbose:
            print(f'Rank {DistMisc.get_rank()} --- Allocated in this process: {allocated_mb:.2f} MB', force=True)
            print(f'Rank {DistMisc.get_rank()} --- Max Allocated in this process: {max_allocated_mb:.2f} MB', force=True)
            print(f'Rank {DistMisc.get_rank()} --- Reserved in this process: {reserved_mb:.2f} MB', force=True)
            print(f'Rank {DistMisc.get_rank()} --- Total: {total_mb:.2f} MB', force=True)
        return allocated_mb, max_allocated_mb, reserved_mb, total_mb
    
    @staticmethod
    def allocate_memory_to_tensor(required_memory_mb, verbose=False, device='cuda'):
        required_memory = required_memory_mb * 1048576
        new_tensor = torch.empty(int(required_memory / 4), dtype=torch.float, device=device)
        if verbose:
            print(f'Rank {DistMisc.get_rank()} --- Now allocated memory: {torch.cuda.memory_allocated() / 1048576:.2f} MB', force=True)
        return new_tensor


class TimeMisc:
    @staticmethod
    def get_time_string():
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


class DummyContextManager:
    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass