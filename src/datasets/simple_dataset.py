import numpy as np
import torch
from torch.utils.data import Dataset

from .modules.data_module_base import DataModuleBase, register


class SimpleDataset(Dataset):
    def __init__(self, X, y, data_form):
        super().__init__()
        self.X_tensor = torch.as_tensor(X, dtype=torch.float32)
        self.y_tensor = torch.as_tensor(y, dtype=torch.float32)
        self.data_form = data_form

    def __getitem__(self, idx):
        X = self.X_tensor[idx]
        gt_y = self.y_tensor[idx]
        
        if self.data_form == '2d':
            return {
                'inputs': {
                    'x': torch.rand((3, 32, 32)),
                },
                'targets': {
                    'gt_y': torch.rand((3, 32, 32)),
                    'index_string': str(idx),
                },
            }
        elif self.data_form == '3d':
            return {
                'inputs': {
                    'x': torch.rand((3, 256, 32, 32)),
                },
                'targets': {
                    'gt_y': torch.rand((3, 256, 32, 32)),
                    'index_string': str(idx),
                },
            }
        else:
            return {
                'inputs': {
                    'x': X,
                },
                'targets': {
                    'gt_y': gt_y,
                    'index_string': str(idx),
                },
            }

    def __len__(self):
        return len(self.y_tensor)


@register('simple')
class SimpleDataModule(DataModuleBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.train_val_len = 10000
        self.test_len = 1000
        if cfg.model.architecture == 'simple_unet2d':
            self.data_form = '2d'
        elif cfg.model.architecture == 'simple_unet3d':
            self.data_form = '3d'
        else:
            self.data_form = 'default'
        
        self.X_train_and_val = np.random.rand(self.train_val_len, 2)
        self.y_train_and_val = np.random.rand(self.train_val_len, 1)
        self.X_test = np.random.rand(self.test_len, 2)
        self.y_test = np.random.rand(self.test_len, 1)
        
    def build_train_dataset(self):
        X = np.random.rand(self.train_val_len, 2)
        y = np.random.rand(self.train_val_len, 1)
        
        n = int(len(y) * self.cfg.data.split_rate)
        X_train = X[:n]
        y_train = y[:n]
        
        return SimpleDataset(X_train, y_train, self.data_form)
        
    def build_val_dataset(self):
        X = np.random.rand(self.train_val_len, 2)
        y = np.random.rand(self.train_val_len, 1)
        
        n = int(len(y) * self.cfg.data.split_rate)
        X_val = X[n:]
        y_val = y[n:]
        
        return SimpleDataset(X_val, y_val, self.data_form)
        
    def build_test_dataset(self):            
        X = np.random.rand(self.test_len, 2)
        y = np.random.rand(self.test_len, 1)
        
        return SimpleDataset(X, y, self.data_form)
