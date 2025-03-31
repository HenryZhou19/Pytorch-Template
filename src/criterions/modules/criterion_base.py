import math

from torch import nn

from src.utils.misc import DistMisc, LoggerMisc
from src.utils.register import Register

criterion_register = Register('criterion')

class CriterionBase(nn.Module):
    registered_name: str
    
    def __init__(self, cfg):
        super().__init__()
        self.ema_mode = False
        self.infer_mode = False
        self.cfg = cfg
        self.loss_config = cfg.criterion.loss
        self.primary_criterion = cfg.criterion.primary_criterion
        if self.primary_criterion is None:
            assert len(self.cfg.trainer.name_optimizers) == 1, 'Main optimizer config must be the only one if primary_criterion is not specified.'
            self.primary_criterion = 'loss_main'
        print(LoggerMisc.block_wrapper(f'primary_criterion: {self.primary_criterion}'))
        
        if cfg.model.ema.ema_enabled and cfg.model.ema.ema_primary_criterion:
            self.primary_criterion = 'ema_' + self.primary_criterion  # use 'ema_xxx' as primary criterion
            
        if cfg.criterion.primary_criterion_higher_better:
            self.choose_better_fn = lambda now, stored: now > stored or math.isnan(stored)  # higher better  
        else:
            self.choose_better_fn = lambda now, stored: now < stored or math.isnan(stored)  # lower better
            
    def set_ema_mode(self, ema_mode):
        self.ema_mode = ema_mode
    
    def set_infer_mode(self, infer_mode):
        self.infer_mode = infer_mode
        
    def print_states(self, prefix='', force=True):
        print(f'{prefix}Criterion --- training mode: {self.training}, infer_mode: {self.infer_mode}, ema_mode: {self.ema_mode}', force=force)
            
    def untrainable_check(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        assert len(trainable_params) == 0, f'Criterion {self.__class__} has trainable parameters.'

    def choose_best(self, last_metric: dict, best_metric: dict):
        # return: new_best_metric, FLAG[last_is_better]
        def compare_primary_criterion(l_m, b_m):
            assert self.primary_criterion in l_m, f'best_criterion "{self.primary_criterion}" not in metric {l_m}'
            if self.choose_better_fn(l_m[self.primary_criterion], b_m[self.primary_criterion]):
                return l_m, True
            else:
                return b_m, False

        assert last_metric != {}, f'last_metric is empty.'
        if best_metric == {}:
            return last_metric, True
        
        return compare_primary_criterion(last_metric, best_metric)
        
    def forward(self, outputs, targets, *args, **kwargs):
        """
        outputs: dict
        targets: dict
        return 
            loss (reduction as mean!), 
            metrics_dict as {
                'loss1': loss1,
                'loss2': loss2,
                'metric1': metric1,
                ...}
            
        Maybe differ in 
            1. self.training=True [train]
            2. self.training=False [eval]
            3. self.training=False and self.infer_mode=True [test/infer]
        """
        if self.infer_mode:
            assert self.training == False, f'CriterionModule {self.__class__} is in training mode while infer_mode is True.'
            
        loss_dict, metrics_dict = self._get_iter_loss_and_metrics(outputs, targets, *args, **kwargs)
        
        if self.ema_mode:
            loss_dict = LoggerMisc.set_dict_key_prefix(loss_dict, 'ema_')
            metrics_dict = LoggerMisc.set_dict_key_prefix(metrics_dict, 'ema_')
            
        return loss_dict, metrics_dict
    
    def forward_epoch_metrics(self):
        epoch_metrics_dict = self._get_epoch_metrics_and_reset()
        
        if self.ema_mode:
            epoch_metrics_dict = LoggerMisc.set_dict_key_prefix(epoch_metrics_dict, 'ema_')
            
        return epoch_metrics_dict
        
    def _get_iter_loss_and_metrics(self, outputs, targets, *args, **kwargs):
        """
        calculate loss and metrics for one iteration
        """
        raise NotImplementedError
        
    def _get_epoch_metrics_and_reset(self):
        """
        metrics which should be calculated after a whole epoch
        """
        return {}
    
    def _if_gather_epoch_metrics(self):
        if_real_dist = self.training or self.infer_mode or self.cfg.trainer.dist_eval
        return DistMisc.is_dist_avail_and_initialized() and if_real_dist