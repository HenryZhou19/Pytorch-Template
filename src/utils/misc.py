import importlib
import logging
import math
import os
import pickle
import random
import shutil
import signal
import sys
import time
import warnings
from argparse import Namespace
from copy import deepcopy
from glob import glob
from math import inf

import numpy as np
import sacred
import torch
import torch.distributed as dist
import wandb
import yaml
from tqdm import tqdm

from .scheduler.warmup_scheduler import (WarmUpCosineAnnealingLR,
                                         WarmupLinearLR, WarmupMultiStepLR)


class ConfigMisc:
    @staticmethod
    def get_configs(config_dir, default_config_name):
        config_path = ConfigMisc._get_config_file_path(config_dir, default_config_name)
        return ConfigMisc._get_configs_from_sacred(config_path)
    
    @staticmethod
    def _get_config_file_path(config_dir, default_config_name):
        args = sys.argv
        config_name = default_config_name
        for arg in args:
            if arg.startswith("config="):
                config_name = arg.split("=")[1]
                break
        return os.path.join(config_dir, config_name + ".yaml")
    
    @staticmethod
    def _get_configs_from_sacred(config_path):
        ex = sacred.Experiment('Config Collector', save_git_info=False)
        ex.add_config(config_path)
        
        def print_sacred_configs(_run):
            final_config = _run.config
            final_config.pop('seed', None)
            config_mods = _run.config_modifications
            config_mods.pop('seed', None)
            print(sacred.commands._format_config(final_config, config_mods))

        @ex.main
        def print_init_config(_config, _run):
            if "RANK" in os.environ:
                rank = int(os.environ["RANK"])
                if rank != 0:
                    return
            print(f"\nInitial configs read by sacred for ALL Ranks:")
            print_sacred_configs(_run)

        config = ex.run_commandline().config
        cfg = ConfigMisc.nested_dict_to_nested_namespace(config)
        if hasattr(cfg, 'seed'):
            delattr(cfg, 'seed')  # seed given by sacred is useless
        
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
            assert not hasattr(ns, n), f'Namespace conflict: {v}'
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
    def read(path):
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f.read())
        return ConfigMisc.nested_dict_to_nested_namespace(config)

    @staticmethod
    def write(path, config, ignore_name_list):
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
        return [str(get_nested_attr(cfg, extra)) for extra in cfg_keys]

    @staticmethod
    def output_dir_extras(cfg):
        extras = '_'.join([cfg.info.start_time] + ConfigMisc.get_specific_list(cfg, cfg.info.name_tags))
        if cfg.special.debug:
            extras = 'debug_' + extras
        return extras


class PortalMisc:
    @staticmethod
    def _find_available_new_path(path, suffix=''):
        if os.path.exists(path):
            counter = 1
            new_path = f"{path}_{suffix}{counter}"
            while os.path.exists(new_path):
                new_path = f"{path}_{suffix}{counter}"
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
            destination_dir = PortalMisc._find_available_new_path(destination_dir, suffix='resume_')
            
            for source_path in source_paths:
                if os.path.isdir(source_path):
                    shutil.copytree(source_path, os.path.join(destination_dir, os.path.basename(source_path)),
                                    ignore=shutil.ignore_patterns('__pycache__'))
                elif os.path.isfile(source_path):
                    shutil.copy(source_path, destination_dir)
                else:
                    print(f"Skipping {source_path} as it is neither a file nor a directory.")
            print(StrMisc.block_wrapper('Files and folders of currect project are copied successfully.'))
    
    @staticmethod
    def combine_train_infer_configs(infer_cfg, use_train_seed=True):
        cfg = ConfigMisc.read(infer_cfg.tester.train_cfg_path)  # config in training
        train_seed = cfg.env.seed
        ConfigMisc.update_nested_namespace(cfg, infer_cfg)
        if use_train_seed:
            cfg.env.seed = train_seed

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
        if cfg.trainer.resume is not None:  # read "work_dir", "start_time" from the .yaml file
            print('Resuming from: ', cfg.trainer.resume, ', reading configs from .yaml file...')
            cfg_old = ConfigMisc.read(cfg.trainer.resume)
            # TODO: assert critial params are the same, but others can be changed(e.g. info...)
            work_dir = cfg_old.info.work_dir
            setattr(cfg.info, 'resume_start_time', cfg.info.start_time)
            cfg.info.start_time = cfg_old.info.start_time
        else:
            work_dir = os.path.join(cfg.info.output_dir, ConfigMisc.output_dir_extras(cfg))
            if DistMisc.is_main_process():
                print('New start at: ', work_dir)
                if not os.path.exists(work_dir):
                    os.makedirs(work_dir)
        cfg.info.work_dir = work_dir

    @staticmethod
    def seed_everything(cfg):
        assert hasattr(cfg.env, 'distributed')
        if cfg.env.seed_with_rank:
            cfg.env.seed = cfg.env.seed + DistMisc.get_rank()
        
        os.environ['PYTHONHASHSEED'] = str(cfg.env.seed)

        random.seed(cfg.env.seed)
        np.random.seed(cfg.env.seed)
        
        torch.manual_seed(cfg.env.seed)
        torch.cuda.manual_seed(cfg.env.seed)
        torch.cuda.manual_seed_all(cfg.env.seed)

        if cfg.env.cuda_deterministic:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    @staticmethod
    def special_config_adjustment(cfg):
        if cfg.special.debug:  # debug mode
            cfg.env.num_workers = 0
        if cfg.trainer.grad_accumulation > 1:
            warnings.warn('Gradient accumulation is set to N > 1. This may affect the function of some modules(e.g. batchnorm, lr_scheduler).')
        cfg.data.batch_size_total = cfg.data.batch_size_per_rank * cfg.env.world_size * cfg.trainer.grad_accumulation

    @staticmethod
    def save_configs(cfg):
        if DistMisc.is_main_process():
            if not os.path.exists(cfg.info.work_dir):
                os.makedirs(cfg.info.work_dir)
            if cfg.trainer.resume is None:
                cfg_file_name = 'cfg.yaml'
            else:
                cfg_file_name = f'cfg_resume_{cfg.info.resume_start_time}.yaml'  
            
            ConfigMisc.write(os.path.join(cfg.info.work_dir, cfg_file_name), cfg, ignore_name_list=cfg.special.print_save_config_ignore)
            
        if cfg.special.save_current_project:
            PortalMisc._save_currect_project(cfg)

    @staticmethod
    def print_config(cfg, force_all_rank=False):  
        def write_msg_lines(msg_in, cfg_in, indent=1):
            for name in sorted(vars(cfg_in).keys()):
                if name in cfg.special.print_save_config_ignore:
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
                    msg_in += f'{m_indent:40}{v}\n'
            return msg_in

        msg = f"Rank {DistMisc.get_rank()} --- Parameters:\n"
        msg = StrMisc.block_wrapper(write_msg_lines(msg, cfg), s='=', block_width=80)

        DistMisc.avoid_print_mess()
        if cfg.env.distributed:
            print(msg, force=force_all_rank)
        else:
            print(msg)
        DistMisc.avoid_print_mess()

    @staticmethod 
    def init_loggers(cfg):
        if DistMisc.is_main_process():
            wandb_name = '_'.join(ConfigMisc.get_specific_list(cfg, cfg.info.name_tags))
            wandb_name = f'[{cfg.info.task_type}] ' + wandb_name
            wandb_tags = ConfigMisc.get_specific_list(cfg, cfg.info.wandb_tags)
            if TesterMisc.is_inference(cfg):
                wandb_tags.append(f'Infer: {cfg.info.infer_start_time}')
            if cfg.trainer.resume != None:
                wandb_tags.append(f'Re: {cfg.info.resume_start_time}')
                if cfg.info.wandb_resume_enabled:
                    resumed_wandb_id = glob(cfg.info.work_dir + '/wandb/latest-run/*.wandb')[0].split('-')[-1].split('.')[0]
            cfg.info.wandb_run = wandb.init(
                project=cfg.info.project_name,
                name=wandb_name,
                tags=wandb_tags,
                dir=cfg.info.work_dir,
                config=ConfigMisc.nested_namespace_to_plain_namespace(cfg, cfg.special.wandb_config_ignore),
                resume='allow' if cfg.trainer.resume and cfg.info.wandb_resume_enabled else None,
                id=resumed_wandb_id if cfg.trainer.resume and cfg.info.wandb_resume_enabled else None,
                )
            cfg.info.log_file = open(os.path.join(cfg.info.work_dir, 'logs.txt'), 'a' if cfg.trainer.resume is None else 'a+')
        else:
            cfg.info.log_file = sys.stdout

    @staticmethod 
    def end_everything(cfg, end_with_printed_cfg=False, force=False):
        if end_with_printed_cfg:
            PortalMisc.print_config(cfg)
        if DistMisc.is_main_process():
            cfg.info.log_file.close()
            print('log_file closed.')
            try:
                if force:
                    wandb.finish(exit_code=-1)
                    print('wandb closed.')
                    exit(0)  # 0 for shutting down bash master_port sweeper
                else:
                    if not cfg.special.debug:
                        for _ in tqdm(range(cfg.info.wandb_buffer_time), desc='Waiting for wandb to upload all files...'):
                            time.sleep(1)
                    wandb.finish()
                    print('wandb closed.')
            finally:
                pass

    @staticmethod 
    def interrupt_handler(cfg):
        """Handles SIGINT signal (Ctrl+C) by exiting the program gracefully."""
        def signal_handler(sig, frame):
            print('Received SIGINT. Cleaning up...')
            PortalMisc.end_everything(cfg, force=True)

        signal.signal(signal.SIGINT, signal_handler)


class DistMisc:
    @staticmethod
    def avoid_print_mess():
        if DistMisc.is_dist_avail_and_initialized():  # 
            dist.barrier()
            time.sleep(DistMisc.get_rank() * 0.1)
    
    @staticmethod
    def all_gather(data):

        """
        Run all_gather on arbitrary picklable data (not necessarily tensors)
        Args:
            data: any picklable object
        Returns:
            list[data]: list of data gathered from each rank
        """
        world_size = DistMisc.get_world_size()
        if world_size == 1:
            return [data]

        # serialized to a Tensor
        buffer = pickle.dumps(data)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage).to("cuda")

        # obtain Tensor size of each rank
        local_size = torch.tensor([tensor.numel()], device="cuda")
        size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        size_list = [int(size.item()) for size in size_list]
        max_size = max(size_list)

        # receiving Tensor from all ranks
        # we pad the tensor because torch all_gather does not support
        # gathering tensors of different shapes
        tensor_list = [None] * world_size
        # tensor_list = []
        # for _ in size_list:
        #     tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
        if local_size != max_size:
            padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
            tensor = torch.cat((tensor, padding), dim=0)
        dist.all_gather(tensor_list, tensor)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))

        return data_list

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
    def reduce_sum(tensor):
        world_size = DistMisc.get_world_size()
        if world_size < 2:
            return tensor
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    @staticmethod
    def reduce_mean(tensor):
        world_size = DistMisc.get_world_size()
        total = DistMisc.reduce_sum(tensor)
        return total.float() / world_size

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
            force = kwargs.pop("force", False)
            if is_master or force:
                builtin_print(*args, **kwargs)

        __builtin__.print = dist_print

    @staticmethod
    def init_distributed_mode(cfg):
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            cfg.env.rank = int(os.environ["RANK"])
            cfg.env.world_size = int(os.environ["WORLD_SIZE"])
            cfg.env.gpu = int(os.environ["LOCAL_RANK"])
        elif "SLURM_PROCID" in os.environ and 'SLURM_PTY_PORT' not in os.environ:
            cfg.env.rank = int(os.environ["SLURM_PROCID"])
            cfg.env.gpu = cfg.env.rank % torch.cuda.device_count()
        else:
            raise NotImplementedError("Must use distributed mode")
            
            # print("Not using distributed mode")
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
        # print(f"INFO - distributed init (Rank {cfg.env.rank}): {cfg.env.dist_url}")
        # DistMisc.avoid_print_mess()
        DistMisc.setup_for_distributed(cfg.env.rank == 0)


class ModelMisc:
    @staticmethod
    def print_model_info(cfg, model, *args):
        if DistMisc.is_main_process():
            args = set(args)
            if len(args) > 0:
                print_str = ''
                if 'model_structure' in args:
                    print_str += str(model) + '\n'
                if 'trainable_params' in args:
                    print_str += f'Trainable parameters: {sum(map(lambda p: p.numel() if p.requires_grad else 0, model.parameters()))}\n'
                if 'total_params' in args:
                    print_str += f'Total parameters: {sum(map(lambda p: p.numel(), model.parameters()))}\n'
                
                print_str = StrMisc.block_wrapper(print_str, s='-', block_width=80)
                print(print_str)
                print(print_str, file=cfg.info.log_file)
                cfg.info.log_file.flush()
            
    @staticmethod
    def print_model_info_with_torchinfo(cfg, model, train_loader, device, info_columns=None):
        if DistMisc.is_main_process():
            import torchinfo
            info_columns = info_columns if info_columns is not None else [
                'input_size',
                'output_size',
                'num_params',
                'params_percent',
                'kernel_size',
                'mult_adds',
                'trainable',
                ]
            input_data = train_loader.dataset.__getitem__(0)['inputs']
            input_data = TensorMisc.to({k: v.unsqueeze(0).expand(cfg.data.batch_size_per_rank, *v.shape) for k, v in input_data.items()}, device)
            assert cfg.data.batch_size_per_rank == train_loader.batch_size
            print_str = torchinfo.summary(model, input_data=input_data, col_names=info_columns, depth=9, verbose=0)
            # Check model info in OUTPUT_PATH/logs.txt
            print(print_str, file=cfg.info.log_file)
            cfg.info.log_file.flush()
    
    @staticmethod
    def ddp_wrapper(cfg, model_without_ddp):
        if cfg.env.distributed:
            model_without_ddp = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_without_ddp)
            
            if cfg.deepspeed.ds_enable:
                return ModelMisc._deepspeed_ddp_wrapper(cfg, model_without_ddp)
            else:
                return torch.nn.parallel.DistributedDataParallel(model_without_ddp, device_ids=[cfg.env.gpu],
                find_unused_parameters=cfg.env.find_unused_params,
            )
        else:
            return model_without_ddp
    
    @staticmethod
    def _deepspeed_ddp_wrapper(cfg, model_without_ddp):
        assert cfg.env.distributed, 'DeepSpeed DDP wrapper only works in distributed mode.'
        
        print(StrMisc.block_wrapper('Using DeepSpeed DDP wrapper...', s='#', block_width=80))
        DistMisc.avoid_print_mess()
        
        import deepspeed
        deepspeed.logger.setLevel(logging.WARNING)
        
        def ds_init_engine_wrapper(model_without_ddp) -> deepspeed.DeepSpeedEngine:          
            return deepspeed.initialize(model=model_without_ddp, config=deepspeed_config)[0]
        
        # with open(cfg.deepspeed.deepspeed_config, 'r') as json_file:
        #     deepspeed_config = hjson.load(json_file)
        deepspeed_config = {}
        deepspeed_config.update({'train_batch_size': cfg.data.batch_size_total})
        deepspeed_config.update({'gradient_accumulation_steps': 1})  # deepspeed will not handle gradient accumulation operations (manually do this in trainer, so keep it '1')
        deepspeed_config.update({"zero_optimization": {"stage": 0}})
        return ds_init_engine_wrapper(model_without_ddp)


class OptimizerMisc:
    @staticmethod
    def _get_param_dicts_with_specific_lr(cfg, model_without_ddp: torch.nn.Module):
        def match_name_keywords(name, name_keywords):
            for keyword in name_keywords:
                if keyword in name:
                    return True
            return False

        if not hasattr(cfg.trainer.optimizer, 'lr'):
            assert hasattr(cfg.trainer.optimizer, 'lr_groups')
            param_dicts_with_lr = [
                {"params": [p for n, p in model_without_ddp.named_parameters()
                            if not match_name_keywords(n, 'backbone') and p.requires_grad],
                "lr": cfg.trainer.optimizer.lr_groups.main,},
                {"params": [p for n, p in model_without_ddp.named_parameters()
                            if match_name_keywords(n, 'backbone') and p.requires_grad],
                "lr": cfg.trainer.optimizer.lr_groups.backbone},
                ]
        else:  # if cfg.trainer.lr exists, then all params use cfg.trainer.lr
            param_dicts_with_lr = [
                {"params": [p for n, p in model_without_ddp.named_parameters()
                            if p.requires_grad],
                "lr": cfg.trainer.optimizer.lr},
                ]
        
        return param_dicts_with_lr
    
    @staticmethod
    def get_optimizer(cfg, model_without_ddp):
        param_dicts_with_lr = OptimizerMisc._get_param_dicts_with_specific_lr(cfg, model_without_ddp)
        if cfg.trainer.optimizer.optimizer_choice == 'adamw':
            return torch.optim.AdamW(param_dicts_with_lr, weight_decay=cfg.trainer.optimizer.weight_decay)
        else:
            raise ValueError(f'Unknown optimizer choice: {cfg.trainer.optimizer.optimizer_choice}')


class SchudulerMisc:   
    @staticmethod
    def get_warmup_lr_scheduler(cfg, optimizer, scaler, train_loader):
        len_train_loader = len(train_loader)
        if cfg.trainer.scheduler.warmup_steps >= 0:
            warmup_iters = cfg.trainer.scheduler.warmup_steps
        elif cfg.trainer.scheduler.warmup_epochs >= 0:
            warmup_iters = cfg.trainer.scheduler.warmup_epochs * len_train_loader
        else:
            warmup_iters = 0
            
        kwargs = {
            "optimizer": optimizer,
            "scaler": scaler,
            "do_grad_accumulation": cfg.trainer.grad_accumulation > 1,
            "T_max": cfg.trainer.epochs * len_train_loader,
            "T_warmup": warmup_iters,
            "warmup_factor": cfg.trainer.scheduler.warmup_factor,
            "lr_min": cfg.trainer.scheduler.lr_min,
        }
            
        if cfg.trainer.scheduler.scheduler_choice == 'cosine':
            return WarmUpCosineAnnealingLR(**kwargs)
        elif cfg.trainer.scheduler.scheduler_choice == 'linear':
            return WarmupLinearLR(**kwargs)
        elif cfg.trainer.scheduler.scheduler_choice == 'multistep':
            if cfg.trainer.scheduler.lr_milestones_steps is not None:
                step_milestones = cfg.trainer.scheduler.lr_milestones_steps
            elif cfg.trainer.scheduler.lr_milestones_epochs is not None:
                step_milestones = [len_train_loader * lr_milestones_epoch for lr_milestones_epoch in cfg.trainer.scheduler.lr_milestones_epochs]
            else:
                raise ValueError('lr_milestones_steps and lr_milestones_epochs cannot be both None.')
            kwargs.update({
                "step_milestones": step_milestones,
                "gamma": cfg.trainer.scheduler.lr_decay_gamma,
            })    
            return WarmupMultiStepLR(**kwargs)
        else:
            raise ValueError(f'Unknown scheduler choice: {cfg.trainer.scheduler.scheduler_choice}')
    

class TrainerMisc:
    @staticmethod
    def get_pbar(cfg, trainer_status):
        if DistMisc.is_main_process():
            len_train_loader = len(trainer_status['train_loader'])
            len_val_loader = len(trainer_status['val_loader'])
            epoch_finished = trainer_status['start_epoch'] - 1
            train_pbar = tqdm(
                total=cfg.trainer.epochs * len_train_loader if cfg.info.global_tqdm else len_train_loader,
                dynamic_ncols=True,
                colour='green',
                position=0,
                maxinterval=inf,
                initial=epoch_finished * len_train_loader,
            )
            train_pbar.set_description_str('Train')
            print('')
            val_pbar = tqdm(
                total=cfg.trainer.epochs * len_val_loader if cfg.info.global_tqdm else len_val_loader,
                dynamic_ncols=True,
                colour='green',
                position=0,
                maxinterval=inf,
                initial=epoch_finished * len_val_loader,
            )
            val_pbar.set_description_str('Eval ')
            print('')
            trainer_status['train_pbar'] = train_pbar
            trainer_status['val_pbar'] = val_pbar
            
        return trainer_status
    
    @staticmethod
    def resume_training(cfg, trainer_status):
        if cfg.trainer.resume:
            print('Resuming from ', cfg.trainer.resume, ', loading the checkpoint...')
            checkpoint_path = glob(os.path.join(cfg.info.work_dir, 'checkpoint_last_epoch_*.pth'))
            assert len(checkpoint_path) == 1, f'Found {len(checkpoint_path)} checkpoints, please check.'
            checkpoint = torch.load(checkpoint_path[0], map_location='cpu')
            trainer_status['model_without_ddp'].load_state_dict(checkpoint['model'])
            trainer_status['optimizer'].load_state_dict(checkpoint['optimizer'])
            trainer_status['lr_scheduler'].load_state_dict(checkpoint['lr_scheduler'])
            if cfg.env.amp:
                trainer_status['scaler'].load_state_dict(checkpoint['scaler'])
            trainer_status['start_epoch'] = checkpoint['epoch'] + 1
            trainer_status['best_metrics'] = checkpoint.get('best_metrics', {})
            trainer_status['metrics'] = checkpoint.get('last_metrics', {})
            trainer_status['train_iters'] = checkpoint['epoch']*len(trainer_status['train_loader'])
        else:
            print('New trainer.')
        print(f"Start from epoch: {trainer_status['start_epoch']}")
        
        return trainer_status
    
    @staticmethod
    def before_one_epoch(cfg, trainer_status, **kwargs):
        assert 'epoch' in kwargs.keys()
        trainer_status['epoch'] = kwargs['epoch']
        if cfg.env.distributed:
            # shuffle data for each epoch (here needs epoch start from 0)
            trainer_status['train_loader'].sampler_set_epoch(trainer_status['epoch'] - 1)  
        
        dist.barrier()
        if DistMisc.is_main_process():
            if cfg.info.global_tqdm:
                trainer_status['train_pbar'].unpause()
            else :
                trainer_status['train_pbar'].reset()
                trainer_status['val_pbar'].reset()

    @staticmethod
    def after_training_before_validation(cfg, trainer_status, **kwargs):
        LoggerMisc.wandb_log(cfg, 'train_epoch', trainer_status['train_outputs'], trainer_status['train_iters'])

        if DistMisc.is_main_process():
            trainer_status['val_pbar'].unpause()

    @staticmethod
    def after_validation(cfg, trainer_status, **kwargs):
        LoggerMisc.wandb_log(cfg, 'val_epoch', trainer_status['metrics'], trainer_status['train_iters'])
        
        TrainerMisc.save_checkpoint(cfg, trainer_status)
    
    @staticmethod
    def after_all_epochs(cfg, trainer_status, **kwargs):
        if DistMisc.is_main_process():
            trainer_status['train_pbar'].close()
            trainer_status['val_pbar'].close()
    

            
    @staticmethod
    def save_checkpoint(cfg, trainer_status):
        if DistMisc.is_main_process():
            epoch_finished = trainer_status['epoch']
            trainer_status['best_metrics'], save_flag = trainer_status['criterion'].choose_best(
                trainer_status['metrics'], trainer_status['best_metrics']
            )


            save_files = {
                'model': trainer_status['model_without_ddp'].state_dict(),
                'best_metrics': trainer_status['best_metrics'],
                'epoch': epoch_finished,
            }

            if save_flag:
                best = glob(os.path.join(cfg.info.work_dir, 'checkpoint_best_epoch_*.pth'))
                assert len(best) <= 1
                if len(best) == 1:
                    torch.save(save_files, best[0])
                    os.rename(best[0], os.path.join(cfg.info.work_dir, f'checkpoint_best_epoch_{epoch_finished}.pth'))
                else:
                    torch.save(save_files, os.path.join(cfg.info.work_dir, f'checkpoint_best_epoch_{epoch_finished}.pth'))

            if (trainer_status['epoch'] + 1) % cfg.trainer.save_interval == 0:
                save_files.update({
                    'optimizer': trainer_status['optimizer'].state_dict(),
                    'lr_scheduler': trainer_status['lr_scheduler'].state_dict(),
                    'last_metric': trainer_status['metrics']
                })
                if cfg.env.amp and cfg.env.device:
                    save_files.update({
                        'scaler': trainer_status['scaler'].state_dict()
                    })
                last = glob(os.path.join(cfg.info.work_dir, 'checkpoint_last_epoch_*.pth'))
                assert len(last) <= 1
                if len(last) == 1:
                    torch.save(save_files, last[0])
                    os.rename(last[0], os.path.join(cfg.info.work_dir, f'checkpoint_last_epoch_{epoch_finished}.pth'))
                else:
                    torch.save(save_files, os.path.join(cfg.info.work_dir, f'checkpoint_last_epoch_{epoch_finished}.pth'))
    
    class BackwardAndStep:
        def __init__(self, cfg, trainer_status):
            self.trainer_status = trainer_status
            assert cfg.trainer.grad_accumulation > 0 and isinstance(cfg.trainer.grad_accumulation, int), 'grad_accumulation should be a positive integer.'
            self.gradient_accumulation_steps = cfg.trainer.grad_accumulation
            self.do_gradient_accumulation = self.gradient_accumulation_steps > 1
            self.max_grad_norm = cfg.trainer.max_grad_norm
            self.model: torch.nn.Module = trainer_status['model']
            self.optimizer: torch.optim.Optimizer = trainer_status['optimizer']
            self.lr_scheduler: torch.optim.lr_scheduler._LRScheduler = trainer_status['lr_scheduler']
            self.scaler: torch.cuda.amp.GradScaler = trainer_status['scaler']
            self.iter_len = len(trainer_status['train_loader'])
            self.step_count = 0
            
            self.optimizer.zero_grad()
            
        def _backward(self, loss: torch.Tensor):
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
        def _optimize(self):
            if self.scaler is not None:
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()
            self.optimizer.zero_grad()
            
        def __call__(self, loss):
            if not math.isfinite(loss):
                raise ValueError(f'Rank {DistMisc.get_rank()}: Loss is {loss}, stopping training')
            
            if self.do_gradient_accumulation:
                loss /= self.gradient_accumulation_steps  # Assume that all losses are mean-reduction. (Otherwise meaningless)
                self.step_count += 1
            
            self._backward(loss)
            
            if self.do_gradient_accumulation and (self.trainer_status['train_iters'] % self.iter_len != 0):
                if self.step_count % self.gradient_accumulation_steps == 0:
                    self._optimize()
                    self.step_count = 0
            else:
                self._optimize()
            
            self.lr_scheduler.step()  # update special lr_scheduler after each iter


class TesterMisc:
    @staticmethod
    def get_pbar(cfg, tester_status):
        if DistMisc.is_main_process():
            test_pbar = tqdm(
                total=len(tester_status['test_loader']),
                dynamic_ncols=True,
                colour='green',
                position=0,
                maxinterval=inf,
            )
            test_pbar.set_description_str('Test ')
            print('')
            tester_status['test_pbar'] = test_pbar
        
        return tester_status
    
    @staticmethod
    def is_inference(cfg):
        return hasattr(cfg, 'tester')
    
    @staticmethod
    def load_model(cfg, tester_status):
        checkpoint = torch.load(cfg.tester.checkpoint_path, map_location=tester_status['device'])
        tester_status['model_without_ddp'].load_state_dict(checkpoint['model'])
        # print(f'{config.mode} mode: Loading pth from', path)
        print('Loading pth from', cfg.tester.checkpoint_path)
        print('best_trainer_metric', checkpoint.get('best_metric', {}))
        if DistMisc.is_main_process():
            if 'epoch' in checkpoint.keys():
                print('Epoch:', checkpoint['epoch'])
                cfg.info.wandb_run.tags = cfg.info.wandb_run.tags + (f"Epoch: {checkpoint['epoch']}",)
        print('last_trainer_metric', checkpoint.get('last_metric', {}))
        
        return tester_status

    @staticmethod
    def before_inference(cfg, tester_status, **kwargs):
        pass

    @staticmethod
    def after_inference(cfg, tester_status, **kwargs):
        LoggerMisc.wandb_log(cfg,  'infer', tester_status['metrics'], None)
        
        if DistMisc.is_main_process():          
            tester_status['test_pbar'].close()


class LoggerMisc:            
    @staticmethod   
    def wandb_log(cfg, group, output_dict, step):
        if DistMisc.is_main_process():
            for k, v in output_dict.items():
                if k == 'epoch':
                    wandb.log({f'{k}': v}, step=step)  # log epoch without group
                else:
                    wandb.log({f'{group}/{k}': v}, step=step)
                # wandb.log({'output_image': [wandb.Image(trainer_status['output_image'])]})
                # wandb.log({"output_video": wandb.Video(trainer_status['output_video'], fps=30, format="mp4")})


class SweepMisc:
    @staticmethod
    def init_sweep_mode(cfg, portal_fn):
        if cfg.sweep.sweep_enabled:
            if hasattr(cfg, 'trainer'):
                if cfg.trainer.resume is not None:
                    print(StrMisc.block_wrapper('Sweep mode cannot be used with resume in phase of training. Ignoring all sweep configs...', '$'))
                    portal_fn(cfg)
            else:
                assert hasattr(cfg, 'tester'), 'Sweep mode can only be used in phase of training or inference.'
            sweep_cfg_dict = ConfigMisc.nested_namespace_to_nested_dict(cfg.sweep.sweep_params)
            
            from itertools import product
            combinations = [dict(zip(sweep_cfg_dict.keys(), values)) for values in product(*sweep_cfg_dict.values())]
            
            for idx, combination in enumerate(combinations):              
                print(StrMisc.block_wrapper(f'Sweep mode: [{idx + 1}/{len(combinations)}] combinations', s='#', block_width=80))
                
                cfg_now = deepcopy(cfg)
                for chained_k, v in combination.items():
                    k_list = chained_k.split('-')
                    ConfigMisc.setattr_for_nested_namespace(cfg_now, k_list, v)
                portal_fn(cfg_now)
        else:
            portal_fn(cfg)


class StrMisc:
    @staticmethod
    def block_wrapper(input_object, s='=', block_width=80):
        str_input = str(input_object)
        if not str_input.endswith('\n'):
            str_input += '\n'
        return '\n' + s * block_width + '\n' + str_input + s * block_width + '\n'


class TensorMisc:
    @staticmethod
    def to(data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, tuple):
            return tuple(TensorMisc.to(d, device) for d in data)
        elif isinstance(data, list):
            return [TensorMisc.to(d, device) for d in data]
        elif isinstance(data, dict):
            return {k: TensorMisc.to(v, device) for k, v in data.items()}
        else:
            raise TypeError(f'Unknown type: {type(data)}')
        
class ImportMisc:
    @staticmethod
    def import_current_dir_all(current_file, current_module_name):
        current_directory = os.path.dirname(current_file)
        current_file_name = os.path.basename(current_file)
        files = os.listdir(current_directory)
        for file in files:
            if file.endswith(".py") and file != current_file_name:
                module_name = os.path.splitext(file)[0]
                importlib.import_module(f"{current_module_name}.{module_name}")      

class TimeMisc:
    @staticmethod
    def get_time_str():
        return time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
 
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
        def __init__(self, block_name, do_print=True):
            self.block_name = block_name
            self.do_print = do_print
            
        def __enter__(self):
            self.timer = TimeMisc.Timer()

        def __exit__(self, *_):
            if self.do_print:
                m_indent = '    ' + self.block_name
                if len(m_indent) > 40:
                    warnings.warn(f'Block name "{self.block_name}" with indent is too long (>40) to display, please check.')
                if len(m_indent) < 38:
                        m_indent += ' ' + '-' * (38 - len(m_indent)) + ' '
                print(f"{m_indent:40s}elapsed time: {self.timer.info['all']:.4f}")
