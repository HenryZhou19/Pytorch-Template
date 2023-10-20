import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.misc import DistMisc
from src.utils.register import Register

register = Register('data_module')

class DataModuleBase:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def get_train_dataset(self):
        raise NotImplementedError
        
    def get_val_dataset(self):
        raise NotImplementedError
        
    def get_test_dataset(self):            
        raise NotImplementedError
        
    @staticmethod
    def collate_fn(data):
        """
        data: 
            list(
                [0] dict{
                    'a': Tensor,
                    'b': Tensor,
                    'c': dict{
                        'x': Tensor,
                        'y': Tensor
                        }
                    }
                [1] ...
                ), len(data) = batch_size
        """
        batch = dict()
        for k, v in data[0].items():
            if isinstance(v, torch.Tensor):
                # every d in data, get d[k] to form a list, then stack them as batched Tensor 
                batch[k] = torch.stack(list(map(lambda d: d[k], data)), dim=0)
            elif isinstance(v, dict):
                batch[k] = DataModuleBase.collate_fn(list(map(lambda d: d[k], data)))
            else:
                raise NotImplementedError
                
        return batch  # batch: dataloader's output
    
    def get_worker_init_fn(self):
        def _seed_worker(worker_id, rank_seed):    
            worker_seed = rank_seed + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
        return partial(_seed_worker, rank_seed=self.cfg.env.seed_base + self.cfg.env.num_workers * DistMisc.get_rank())
    
    @staticmethod
    def get_generator():
        g = torch.Generator()
        g.manual_seed(0)
        return g
    
    
class DataLoaderX(DataLoader):
    def sampler_set_epoch(self, epoch):
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
            

class InfiniteDataLoaderX(DataLoaderX):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.sampler) if self.batch_sampler is None else len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)