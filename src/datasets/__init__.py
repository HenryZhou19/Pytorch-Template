from torch.utils.data import (Dataset, RandomSampler, Sampler,
                              SequentialSampler, distributed)

from src.utils.misc import ImportMisc

from .modules.data_module_base import DataLoaderX, DataModuleBase, InfiniteDataLoaderX, register

ImportMisc.import_current_dir_all(__file__, __name__)

class DataManager(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.data_module = self._get_data_module()
        self.use_infinite_dataloader = self.cfg.data.infinite_dataloader
        
    def _get_data_module(self):  # DataModule provides methods for getting train/val/test datasets
        data_module: DataModuleBase = register.get(self.cfg.data.dataset)(self.cfg)
        return data_module
        
    def build_dataset(self, split=None) -> DataLoaderX:
        assert split in ['train', 'val', 'test'], f'Invalid split {split}'
        if split == 'train':
            dataset = self.data_module.get_train_dataset()
        elif split == 'val':
            dataset = self.data_module.get_val_dataset()
        else: # split == 'test':
            dataset = self.data_module.get_test_dataset()
        
        dist_sampler = True if split == 'train' else self.cfg.trainer.dist_eval
        dataloader = self._get_dataloader(dataset, dist_sampler=dist_sampler, is_training=split=='train')
        print(f'{split} dataloader built successfully.')
        return dataloader
        
    def _get_dataloader(self, dataset: Dataset, dist_sampler: bool, is_training: bool):
        sampler = self._get_sampler(dataset, is_training, dist_sampler)
        DataloaderClass = InfiniteDataLoaderX if self.use_infinite_dataloader else DataLoaderX
        return DataloaderClass(
            dataset,
            self.cfg.data.batch_size_per_rank,
            sampler=sampler,
            pin_memory=self.cfg.env.pin_memory,
            collate_fn=self.data_module.collate_fn,
            num_workers=self.cfg.env.num_workers,
            worker_init_fn=self.data_module.get_worker_init_fn(),
            generator=self.data_module.get_generator(),
            persistent_workers=True if self.cfg.env.num_workers > 0 else False,
            drop_last=is_training,
        )
        
    def _get_sampler(self, dataset: Dataset, is_training: bool, dist_sampler: bool) -> Sampler:
        if self.cfg.env.distributed and dist_sampler:
            sampler = distributed.DistributedSampler(dataset, shuffle=is_training)
        else:
            if is_training:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
        return sampler
