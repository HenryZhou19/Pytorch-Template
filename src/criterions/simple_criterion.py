from torch import nn

from .modules.criterion_base import CriterionBase
from .modules.criterion_register import register


@register('simple')
class SimpleCriterion(CriterionBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, outputs, targets, mode):
        super().forward(outputs, targets, mode)
        
        pred_y = outputs['pred_y']
        gt_y = targets['gt_y']   

        # metrics (loss) used for backprop
        if self.loss_config == 'mse':
            mse_loss = self.mse_loss(pred_y, gt_y.view_as(pred_y))
            loss = 1 * mse_loss
        else:
            raise NotImplementedError(f'loss "{self.loss_config}" has not been implemented yet.')
    
        # metrics not used for backprop
        pred_y = pred_y.detach()
        gt_y = gt_y.detach()
        l1_loss = self.l1_loss(pred_y, gt_y.view_as(pred_y))

        return loss, {
            'mse_loss': mse_loss,
            'L1_loss': l1_loss,
            }