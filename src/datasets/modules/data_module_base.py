import torch
from torch.utils.data import DataLoader

class DataModuleBase:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def get_train_dataset(self):
        raise NotImplementedError
        
    def get_val_dataset(self):
        raise NotImplementedError
        
    def get_test_dataset(self):            
        raise NotImplementedError
    
    
class DataLoaderX(DataLoader):
    def sampler_set_epoch(self, epoch):
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)


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
            batch[k] = collate_fn(list(map(lambda d: d[k], data)))
            
    return batch  # batch: dataloader's output