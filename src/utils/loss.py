from torch import nn


class LossBase(nn.Module):
    def forward(self, preds, gts):
        # return loss (reduction as mean!), loss_dict as {'loss1': loss, 'loss2': loss2, ...}
        raise NotImplementedError


class SimpleLoss(LossBase):
    def __init__(self, cfg):
        super().__init__()
        self.loss_config = cfg.model.loss
        self.mse_loss = nn.MSELoss()

    def forward(self, preds, gts):
        if self.loss_config == 'mse':
            mse_loss = self.mse_loss(preds, gts)
            loss = 1 * mse_loss
            return loss, {
                'mse_loss': mse_loss
            }
        else:
            raise NotImplementedError(f'loss "{self.loss_config}" has not been implemented yet.')
