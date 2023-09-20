
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .dataset import SimpleDataModule, collate_fn


class DataLoaderX(DataLoader):
    def sampler_set_epoch(self, epoch):
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)


class DataManager(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.data_module = SimpleDataModule(cfg)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def build_dataset(self, split=None, shuffle=False) -> DataLoaderX:
        assert split in ['train', 'val', 'test'], f'Invalid split {split}'
        if split == 'train':
            if self.train_dataset is None:
                self.train_dataset, self.val_dataset = self.data_module.get_train_and_val_dataset()
            dataset = self.train_dataset
        elif split == 'val':
            if self.val_dataset is None:
                _, self.val_dataset = self.data_module.get_train_and_val_dataset()
            dataset = self.val_dataset
        else: # split == 'test':
            if self.test_dataset is None:
                self.test_dataset = self.data_module.get_test_dataset()
            dataset = self.test_dataset
            
        dataloader = self._get_dataloader(dataset, shuffle=shuffle)
        print(f'{split} dataloader built successfully.')
        return dataloader
        
    def _get_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoaderX:
        sampler = self._get_sampler(dataset, shuffle)
            
        return DataLoaderX(
            dataset,
            self.cfg.data.batch_size_per_rank,
            sampler=sampler,
            # pin_memory=True,
            collate_fn=collate_fn,
            num_workers=self.cfg.env.num_workers,
            persistent_workers=True if self.cfg.env.num_workers > 0 else False,
        )
        
    def _get_sampler(self, dataset: Dataset, shuffle: bool) -> Sampler:
        if self.cfg.env.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        else:
            if shuffle:
                sampler = torch.utils.data.RandomSampler(dataset)
            else:
                sampler = torch.utils.data.SequentialSampler(dataset)
        return sampler
