from torch import nn

from src.utils.register import Register

register = Register('criterion')

class CriterionBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.loss_config = cfg.criterion.loss
        self.primary_criterion = cfg.criterion.primary_criterion
        if cfg.criterion.primary_criterion_higher_better:
            self.choose_better_fn = lambda now, stored: now > stored  # higher better  
        else:
            self.choose_better_fn = lambda now, stored: now < stored  # lower better

    def choose_best(self, metric: dict, best_metric: dict):
        def compare_loss(m, b_m):
            if m['loss'] < b_m['loss']:
                return m, True
            else:
                return b_m, False
            
        def compare_primary_criterion(m, b_m):
            assert self.primary_criterion in m, f'best_criterion "{self.primary_criterion}" not in metric {m}'
            if self.choose_better_fn(m[self.primary_criterion], b_m[self.primary_criterion]):
                return m, True
            else:
                return b_m, False

        if metric == {} or best_metric == {}:
            return metric, True
        elif self.primary_criterion is not None:
            return compare_primary_criterion(metric, best_metric)
        else:
            return compare_loss(metric, best_metric)
        
    def forward(self, outputs, targets, mode):
        if mode == 'train':
            pass
        elif mode == 'eval':
            pass
        elif mode == 'test':
            pass
        else:
            raise NotImplementedError(f'mode "{mode}" has not been implemented yet.')
        """
        return 
            loss (reduction as mean!), 
            metrics_dict as {
                'loss1': loss1,
                'loss2': loss2,
                'metric1': metric1,
                ...}
        """
        