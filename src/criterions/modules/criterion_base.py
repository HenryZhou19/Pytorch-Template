from torch import nn

from src.utils.register import Register

criterion_register = Register('criterion')

class CriterionBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.loss_config = cfg.criterion.loss
        self.primary_criterion = cfg.criterion.primary_criterion
        if self.primary_criterion is None:
            self.primary_criterion = 'loss'
        
        if cfg.model.ema.ema_enabled and cfg.model.ema.ema_primary_criterion:
            self.primary_criterion = 'ema_' + self.primary_criterion  # use 'ema_xxx' as primary criterion
            
        if cfg.criterion.primary_criterion_higher_better:
            self.choose_better_fn = lambda now, stored: now > stored  # higher better  
        else:
            self.choose_better_fn = lambda now, stored: now < stored  # lower better
            
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
        
    def forward(self, outputs, targets, infer_mode=False):
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
            3. self.training=False and infer_mode=True [test/infer]
        """
        if infer_mode:
            assert self.training == False, f'CriterionModule {self.__class__} is in training mode while infer_mode is True.'
        
    def get_epoch_metrics_and_reset(self):
        """
        metrics which should be calculated after a whole epoch
        """
        return {}