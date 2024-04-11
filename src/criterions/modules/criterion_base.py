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

    def choose_best(self, metric: dict, best_metric: dict):
          
        def compare_primary_criterion(m, b_m):
            assert self.primary_criterion in m, f'best_criterion "{self.primary_criterion}" not in metric {m}'
            if self.choose_better_fn(m[self.primary_criterion], b_m[self.primary_criterion]):
                return m, True
            else:
                return b_m, False

        if metric == {} or best_metric == {}:
            return metric, True
        
        return compare_primary_criterion(metric, best_metric)
        
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