from torch import nn


class MetricBase:
    def __init__(self, cfg):
        self.primary_criterion = cfg.model.primary_criterion
    
    def get_metrics(self, cfg):
        raise NotImplementedError

    def choose_best(self, metric: dict, best_metric: dict):
        def compare_loss(m, b_m):
            if m['loss'] < b_m['loss']:
                return m, True
            else:
                return b_m, False
            
        def compare_primary_criterion(m, b_m):
            assert self.primary_criterion in m, f'best_criterion "{self.primary_criterion}" not in metric {m}'
            if m[self.primary_criterion] < b_m[self.primary_criterion]:
                return m, True
            else:
                return b_m, False

        if metric == {} or best_metric == {}:
            return metric, True
        elif self.primary_criterion is not None:
            return compare_primary_criterion(metric, best_metric)
        else:
            return compare_loss(metric, best_metric)
        

class SimpleMetric(MetricBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.l1_loss = nn.L1Loss()
    
    def get_metrics(self, preds, gts):
        l1_loss = self.l1_loss(preds, gts)
        return {
            'L1_loss': l1_loss
            }, None