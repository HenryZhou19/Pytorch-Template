import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST

from .modules.data_module_base import DataModuleBase, data_module_register


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


@data_module_register('simple')
class SimpleDataModule(DataModuleBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.train_val_len = 10000
        self.test_len = 1000
        if cfg.model.model_choice == 'simple_unet2d':
            self.data_form = '2d'
        elif cfg.model.model_choice == 'simple_unet3d':
            self.data_form = '3d'
        else:
            self.data_form = 'default'
            
    def _get_random_X_y(self, length):
        seed_used = np.random.get_state()[1][0]
        np.random.seed(0)
        X = np.random.rand(length, 2)
        y = np.random.rand(length, 1)
        np.random.seed(seed_used)
        return X, y
        
    def build_train_dataset(self):
        X, y = self._get_random_X_y(self.train_val_len)
        
        n = int(len(y) * self.cfg.data.split_rate)
        X_train = X[:n]
        y_train = y[:n]
        
        return SimpleDataset(X_train, y_train, self.data_form)
        
    def build_val_dataset(self):
        X, y = self._get_random_X_y(self.train_val_len)
        
        n = int(len(y) * self.cfg.data.split_rate)
        X_val = X[n:]
        y_val = y[n:]
        
        return SimpleDataset(X_val, y_val, self.data_form)
        
    def build_test_dataset(self):
        X, y = self._get_random_X_y(self.test_len)
        
        return SimpleDataset(X, y, self.data_form)
    
    
class MNISTX(MNIST):
    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        return {
            'inputs': {
                'x': img.numpy(),
            },
            'targets': {
                'gt_y': target,
            },
        }
    

@data_module_register('mnist')
class MnistDataModule(DataModuleBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        import torchvision.transforms as transforms

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.MINST = MNISTX
        
    def build_train_dataset(self):
        return self.MINST(root='./data', train=True, transform=self.transform, download=True)
        
    def build_val_dataset(self):
        return self.MINST(root='./data', train=False, transform=self.transform)
        
    def build_test_dataset(self):
        print('Test dataset is just the same as val dataset!')
        return self.MINST(root='./data', train=False, transform=self.transform)

