from src.utils.misc import ImportMisc

from .modules.data_module_base import DataLoaderX, DataModuleBase, register

ImportMisc.import_current_dir_all(__file__, __name__)

class DataManager(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.data_module = self._get_data_module()
        
    def _get_data_module(self):  # DataModule provides methods for getting train/val/test datasets
        data_module: DataModuleBase = register.get(self.cfg.data.dataset)(self.cfg)
        return data_module
        
    def build_dataloader(self, split) -> DataLoaderX:
        dataloader = self.data_module.get_dataloader(split)
        print(f'{split} dataloader built successfully.')
        return dataloader
    