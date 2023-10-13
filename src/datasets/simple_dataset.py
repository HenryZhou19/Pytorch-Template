import numpy as np
import torch
from torch.utils.data import Dataset

from .modules.data_module_base import DataModuleBase
from .modules.data_module_register import register


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X_tensor = torch.tensor(X, dtype=torch.float32)
        self.y_tensor = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, idx):
        X = self.X_tensor[idx]
        gt_y = self.y_tensor[idx]

        return {
            'inputs': {
                'x': X,
            },
            'targets': {
                'gt_y': gt_y,
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
        
        self.X_train_and_val = np.random.rand(self.train_val_len, 2)
        self.y_train_and_val = np.random.rand(self.train_val_len, 1)
        self.X_test = np.random.rand(self.test_len, 2)
        self.y_test = np.random.rand(self.test_len, 1)
        
    def get_train_dataset(self):
        X = np.random.rand(self.train_val_len, 2)
        y = np.random.rand(self.train_val_len, 1)
        
        n = int(len(y) * self.cfg.data.split_rate)
        X_train = X[:n]
        y_train = y[:n]
        
        return SimpleDataset(X_train, y_train)
        
    def get_val_dataset(self):
        X = np.random.rand(self.train_val_len, 2)
        y = np.random.rand(self.train_val_len, 1)
        
        n = int(len(y) * self.cfg.data.split_rate)
        X_val = X[n:]
        y_val = y[n:]
        
        return SimpleDataset(X_val, y_val)
        
    def get_test_dataset(self):            
        X = np.random.rand(self.test_len, 2)
        y = np.random.rand(self.test_len, 1)
        
        return SimpleDataset(X, y)
