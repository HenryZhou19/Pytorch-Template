import itertools
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import (DataLoader, Dataset, RandomSampler, Sampler,
                              SequentialSampler, distributed)

from src.utils.misc import DistMisc, TensorMisc
from src.utils.register import Register

data_module_register = Register('data_module')

class DataModuleBase:
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
        AcceptableType: torch.Tensor, str, dict, int, float, bool, np.ndarray
        data: 
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
                # every d in data, get d[k]: Tensor to form a list, then stack them as a batched Tensor
                batch[k] = torch.stack(list(map(lambda d: d[k], data)), dim=0)
            elif isinstance(v, np.ndarray):
                # every d in data, get d[k]: ndarray to form a list of Tensors, then stack them as a batched Tensor
                batch[k] = torch.stack(list(map(lambda d: torch.as_tensor(d[k]), data)), dim=0)
            elif isinstance(v, (int, float, bool)):
                # every d in data, get d[k]: (int, float, bool) to form a batched Tensor
                batch[k] = torch.as_tensor(list(map(lambda d: d[k], data)))
            elif isinstance(v, str):
                # every d in data, get d[k]: str to form a list, which will not be on cuda later
                batch[k] = TensorMisc.NotToCudaList(map(lambda d: d[k], data))
            else:
                raise NotImplementedError(f'collate_fn not implemented for {type(v)}')
        if not recursion:
            batch['batch_size'] = len(data)
        return batch  # batch: dataloader's output
    
    def get_dataloader(self, split: str):
        assert split in ['train', 'val', 'test'], f'Invalid split {split}'
        dataset = self.get_dataset(split)
        
        is_train = split=='train'
        is_train_or_val = split in ['train', 'val']
        use_dist_sampler = True if split == 'train' else self.cfg.trainer.dist_eval
        
        if is_train_or_val and getattr(self.cfg.trainer, 'infinite_dataloader', False):
            DataloaderClass = InfiniteDataLoaderX
        elif is_train and getattr(self.cfg.trainer, 'fixed_length_dataloader', 0) > 0:
            DataloaderClass = partial(
                FixedLengthDataLoaderX,
                total_batches_one_epoch=self.cfg.trainer.fixed_length_dataloader * self.cfg.trainer.grad_accumulation,
                total_epochs=self.cfg.trainer.epochs,
                )
        else:
            DataloaderClass = DataLoaderX
        return DataloaderClass(
            dataset=dataset,
            batch_size=self.cfg.trainer.trainer_batch_size_per_rank if is_train_or_val else self.cfg.tester.tester_batch_size_per_rank,
            sampler=self.get_sampler(dataset, is_train, use_dist_sampler),
            pin_memory=self.cfg.env.pin_memory,
            collate_fn=self.collate_fn,
            num_workers=self.cfg.env.num_workers,
            worker_init_fn=self.get_worker_init_fn(),
            generator=self.get_generator(),
            prefetch_factor=self.cfg.env.prefetch_factor if self.cfg.env.num_workers > 0 else None,
            persistent_workers=True if self.cfg.env.num_workers > 0 else False,
            drop_last=is_train,
        )
    
    def get_worker_init_fn(self):
        def _worker_init_fn(worker_id, rank_seed):
            worker_seed = rank_seed + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
        return partial(_worker_init_fn, rank_seed=self.cfg.seed_base + self.cfg.env.num_workers * DistMisc.get_rank())
    
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
        
    def reinit_batch_size(self, batch_size):
        if 'batch_size' in self.init_kwargs:
            self.init_kwargs['batch_size'] = batch_size
        else:
            raise NotImplementedError('batch_size not in init_kwargs, so cannot reinit_batch_size')
        return DataLoaderX(*self.init_args, **self.init_kwargs)
    
    def sampler_set_epoch(self, epoch):
        self._current_epoch = epoch
        
        if self.sampler is not None:
            self.sampler.set_epoch(self._current_epoch)
            

class InfiniteDataLoaderX(DataLoaderX):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        
        assert self.batch_sampler is not None, 'batch_sampler should not be None'
        self.batch_sampler = _InfiniteSampler(self.batch_sampler)
        
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()
        
    def __len__(self):
        return len(self.batch_sampler.sampler)
        
    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _InfiniteSampler:
    def __init__(self, sampler):
        self.sampler = sampler
        
    def __iter__(self):
        while True:
            yield from iter(self.sampler)
            

class FixedLengthSampler:
    def __init__(self, raw_sampler_list, required_length):
        self.raw_sampler_list = raw_sampler_list
        self.required_length = required_length
        self._check_length()
        
    def _check_length(self):
        total_length = sum(map(len, self.raw_sampler_list))
        assert total_length >= self.required_length, f'total_length {total_length} < required_length {self.required_length}'
        
    def __iter__(self):
        return itertools.islice(itertools.chain(*self.raw_sampler_list), self.required_length)
    
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
        raw_index_sampler = self._raw_index_sampler
        len_raw_index_sampler = len(raw_index_sampler)
        
        raw_index_sampler_list = [raw_index_sampler]
        now_length = len_raw_index_sampler
        
        while now_length < self._total_batches_one_epoch:
            self.sampler_set_epoch(self._current_epoch + self._total_epochs)
            raw_index_sampler_list.append(self._raw_index_sampler)
            now_length += len_raw_index_sampler
    
        return FixedLengthSampler(raw_index_sampler_list, self._total_batches_one_epoch)