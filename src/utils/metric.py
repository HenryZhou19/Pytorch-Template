from torch import nn


class MetricBase:
    def get_metrics(self, preds, gts):
        raise NotImplementedError

    def choose_best(self, metric: dict, best_metric: dict):
        raise NotImplementedError
        

class SimpleMetric(MetricBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.l1_loss = nn.L1Loss()
    
    def get_metrics(self, preds, gts):
        l1_loss = self.l1_loss(preds, gts)
        return {
            'L1_loss': l1_loss
            }, None
            
    def choose_best(self, metric: dict, best_metric: dict):
        def compare_loss(m, b_m):
            if m['loss'] < b_m['loss']:
                return m, True
            else:
                return b_m, False

        if metric == {} or best_metric == {}:
            return metric, True
        else:
            return compare_loss(metric, best_metric)