from torch import nn


class LossBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.loss_config = cfg.criterion.loss
        
    def forward(self, outputs, targets):
        # return loss (reduction as mean!), loss_dict as {'loss1': loss, 'loss2': loss2, ...}
        raise NotImplementedError


class SimpleLoss(LossBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        pred_y = outputs['pred_y']
        gt_y = targets['gt_y']
        if self.loss_config == 'mse':
            mse_loss = self.mse_loss(pred_y, gt_y.view_as(pred_y))
            loss = 1 * mse_loss
            return loss, {
                'mse_loss': mse_loss,
            }
        else:
            raise NotImplementedError(f'loss "{self.loss_config}" has not been implemented yet.')
