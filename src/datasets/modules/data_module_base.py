import itertools
import random
import signal
from functools import partial

import numpy as np
import torch
from torch.utils.data import (DataLoader, Dataset, RandomSampler, Sampler,
                              SequentialSampler, distributed)

from src.utils.misc import DistMisc, TensorMisc
from src.utils.register import Register

data_module_register = Register('data_module')

class DataModuleBase:
    registered_name: str
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def build_train_dataset(self) -> Dataset:
        raise NotImplementedError
        
    def build_val_dataset(self) -> Dataset:
        raise NotImplementedError
        
    def build_test_dataset(self) -> Dataset:            
        raise NotImplementedError
    
    def get_dataset(self, split) -> Dataset:
        if split == 'train':
            if self.train_dataset is None:
                self.train_dataset = self.build_train_dataset()
            return self.train_dataset
        elif split == 'val':
            if self.val_dataset is None:
                self.val_dataset = self.build_val_dataset()
            return self.val_dataset
        elif split == 'test':
            if self.test_dataset is None:
                self.test_dataset = self.build_test_dataset()
            return self.test_dataset
        else:
            raise NotImplementedError(f'Invalid split {split}')
        
    @staticmethod
    def collate_fn(data, recursion=False):
        """
        AcceptableType: dict, torch.Tensor, np.ndarray, int, float, bool, str, tuple, list
        `dict` Type will always be processed recursively.
        `torch.Tensor` and `np.ndarray` will be stacked as a batched ND-Tensor.
        `int`, `float`, `bool` will be stacked as a batched 1D-Tensor.
        `str` will be stacked as a batched list (TensorMisc.NotToCudaBatchList), which will not be on cuda later.
        `list` will be simply stacked as a batched list (to support `Tensor` or `ndarray` of different shapes).
        
        NOTE 1: For `tuple`, please use `list` instead of `tuple` in data as `pin_memory=True` will convert all tuples to lists
        
        NOTE 2: If you are sure that the elements are not needed to be on cuda later, 
            try to use `TensorMisc.NotToCudaBatchList` instead of `list` in `__getitem__` function of your Dataset class,
            This could speedup `TensorMisc.to` a little bit.
        
        data (recursion=False): 
            list(
                [0] dict{
                    'a': AcceptableType,
                    'b': AcceptableType,
                    'c': dict{
                        'x': AcceptableType,
                        'y': dict{...},
                        }
                    }
                [1] ...
                ), len(data) = batch_size       
        which means the '__getitem__' of dataset should return a dict, whose values are AcceptableType
        """
        batch = dict()
        for k, v in data[0].items():
            if isinstance(v, dict):
                # recursion
                batch[k] = DataModuleBase.collate_fn(list(map(lambda d: d[k], data)), recursion=True)
            elif isinstance(v, torch.Tensor):
                # every d in data, get d[k]: `Tensor` to form a list, then stack them as a batched ND-Tensor
                batch[k] = torch.stack(list(map(lambda d: d[k], data)), dim=0)
            elif isinstance(v, np.ndarray):
                # every d in data, get d[k]: `ndarray` to form a list of Tensors, then stack them as a batched ND-Tensor
                batch[k] = torch.stack(list(map(lambda d: torch.as_tensor(d[k]), data)), dim=0)
            elif isinstance(v, (int, float, bool)):
                # every d in data, get d[k]: `(int, float, bool)` to form a batched 1D-Tensor
                batch[k] = torch.as_tensor(list(map(lambda d: d[k], data)))
            elif isinstance(v, (str, TensorMisc.NotToCudaBatchList)):
                # every d in data, get d[k]: `(str, TensorMisc.NotToCudaBatchList)` to form a NotToCudaBatchList, which will not be on cuda later
                batch[k] = TensorMisc.NotToCudaBatchList(map(lambda d: d[k], data))
            elif isinstance(v, list):
                # every d in data, get d[k]: `list` to simply form a BatchList (to support `Tensor` or `ndarray` of different shapes)
                batch[k] = TensorMisc.BatchList(map(lambda d: d[k], data))
            elif isinstance(v, tuple):
                raise TypeError(f'Please use `list` instead of `tuple` in data as `pin_memory=True` will convert all tuples to lists')
            else:
                raise NotImplementedError(f'DataModuleBase.collate_fn not implemented for Type: {type(v)} of Element: {v} with Key: {k}')
        if not recursion:
            batch['batch_size'] = len(data)
        return batch  # batch: dataloader's output
    
    def get_collate_fn(self):
        return self.collate_fn
    
    def get_dataloader(self, split: str):
        assert split in ['train', 'val', 'test'], f'Invalid split {split}'
        is_train = split=='train'
        is_val = split=='val'
        is_test = split=='test'
        if is_train:
            fixed_length_loader = getattr(self.cfg.trainer, 'fixed_length_trainloader', -1)
        elif is_val:
            fixed_length_loader = getattr(self.cfg.trainer, 'fixed_length_valloader', -1)
        else:
            fixed_length_loader = -1  # -1 means normal dataloader, not fixed-length or zero-length dataloader
        
        if fixed_length_loader == 0:  # use negative value to indicate a zero-length dataloader
            assert is_val, 'Only val dataloader can be set to zero-length dataloader'
            return EmptyDataLoader()
        else:
            dataset = self.get_dataset(split)
            use_dist_sampler = True if split == 'train' else self.cfg.trainer.dist_eval
            
            if fixed_length_loader > 0:
                DataloaderClass = partial(
                    FixedLengthDataLoaderX,
                    total_batches_one_epoch=fixed_length_loader,
                    total_epochs=self.cfg.trainer.epochs,
                    )
            else:  # fixed_length_loader < 0
                DataloaderClass = DataLoaderX
                
            batch_size = self.cfg.tester.tester_batch_size_per_rank if is_test else self.cfg.trainer.trainer_batch_size_per_rank
            if split=='val' and self.cfg.special.single_eval:
                batch_size = 1
                
            # check batch_size and dataset length
            if is_train:  # drop_last for training dataloader, so dataset length should be >= batch_size
                if len(dataset) < self.cfg.trainer.trainer_batch_size_total:
                    raise ValueError(f'Dataset length for split {split} ({len(dataset)}) is smaller than the total batch size ({self.cfg.trainer.trainer_batch_size_total}), which may cause dataloader to have 0 batches. Please reduce the batch size or make sure the dataset has enough samples.')
            else:  # not drop_last for val/test dataloader, so dataset length should be > 0
                if len(dataset) == 0:
                    raise ValueError(f'Dataset length for split {split} is 0, which may cause dataloader to have 0 batches. Please make sure the dataset has enough samples.')
            
            return DataloaderClass(
                dataset=dataset,
                batch_size=batch_size,
                sampler=self.get_sampler(dataset, is_train, use_dist_sampler),
                pin_memory=self.cfg.env.pin_memory,
                collate_fn=self.get_collate_fn(),
                num_workers=self.cfg.env.num_workers,
                worker_init_fn=self.get_worker_init_fn(),
                generator=self.get_generator(),
                prefetch_factor=self.cfg.env.prefetch_factor if self.cfg.env.num_workers > 0 else None,
                persistent_workers=True if self.cfg.env.num_workers > 0 else False,
                drop_last=is_train,
            )
    
    @staticmethod
    def _worker_init_fn(worker_id, rank_seed):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        worker_seed = rank_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    def get_worker_init_fn(self):
        return partial(DataModuleBase._worker_init_fn, rank_seed=self.cfg.seed_base + self.cfg.env.num_workers * DistMisc.get_rank())
    
    def get_sampler(self, dataset: Dataset, is_training: bool, use_dist_sampler: bool) -> Sampler:
        if self.cfg.env.distributed and use_dist_sampler:
            sampler = distributed.DistributedSampler(dataset, shuffle=is_training)
        else:
            if is_training:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
        return sampler
    
    @staticmethod
    def get_generator():
        g = torch.Generator()
        g.manual_seed(0)
        return g
    
    
class DataLoaderX(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_args = args
        self.init_kwargs = kwargs
        self._current_epoch = -1
        
    def reinit_batch_size(self, batch_size):
        if 'batch_size' in self.init_kwargs:
            self.init_kwargs['batch_size'] = batch_size
        else:
            raise NotImplementedError('batch_size not in init_kwargs, so cannot reinit_batch_size')
        return DataLoaderX(*self.init_args, **self.init_kwargs)
    
    def sampler_set_epoch(self, epoch):
        self._current_epoch = epoch
        if self.sampler is not None and hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(self._current_epoch)
            

class FixedLengthSampler:
    def __init__(self, raw_index_list, required_length):
        self.raw_index_list = raw_index_list
        self.required_length = required_length
        self._check_length()
        
    def _check_length(self):
        total_length = len(self.raw_index_list)
        assert total_length >= self.required_length, f'total_length {total_length} < required_length {self.required_length}'
        
    def __iter__(self):
        return itertools.islice(iter(self.raw_index_list), self.required_length)
    
    def __len__(self):
        return self.required_length


class FixedLengthDataLoaderX(DataLoaderX):
    def __init__(self, total_batches_one_epoch, total_epochs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._total_batches_one_epoch = total_batches_one_epoch
        self._total_epochs = total_epochs
        
    def __len__(self):
        return self._total_batches_one_epoch
    
    @property
    def _raw_index_sampler(self):
        if self._auto_collation:
            return self.batch_sampler
        else:
            return self.sampler
    
    @property
    def _index_sampler(self):
        len_raw_index_sampler = len(self._raw_index_sampler)
        assert len_raw_index_sampler > 0, f'raw_index_sampler has length 0 (which indicates a dataset smaller than the total batch size over all ranks), thus cannot build the FixedLengthSampler.'
        raw_index_list = list(self._raw_index_sampler)
        now_length = len_raw_index_sampler
        
        while now_length < self._total_batches_one_epoch:
            self.sampler_set_epoch(self._current_epoch + self._total_epochs)
            raw_index_list.extend(list(self._raw_index_sampler))
            now_length += len_raw_index_sampler
    
        return FixedLengthSampler(raw_index_list, self._total_batches_one_epoch)
    

class EmptyDataLoader:
    def __init__(self, *args, **kwargs):
        pass
    
    def __iter__(self):
        return iter([])
    
    def __len__(self):
        return 0
