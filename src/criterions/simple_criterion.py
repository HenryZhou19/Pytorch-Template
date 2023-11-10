from torch import nn

from .modules.criterion_base import CriterionBase, criterion_register


@criterion_register('simple')
class SimpleCriterion(CriterionBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, outputs, targets, infer_mode=False):
        super().forward(outputs, targets, infer_mode)
        
        pred_y = outputs['pred_y']
        gt_y = targets['gt_y']   

        # metrics (loss) used for backprop
        if self.loss_config == 'mse':
            mse_loss = self.mse_loss(pred_y, gt_y.reshape_as(pred_y))
            loss = 1 * mse_loss
        else:
            raise NotImplementedError(f'loss "{self.loss_config}" has not been implemented yet.')
    
        # metrics not used for backprop
        pred_y = pred_y.detach()
        gt_y = gt_y.detach()
        l1_loss = self.l1_loss(pred_y, gt_y.reshape_as(pred_y))

        # if infer_mode:
        #     return None, {
        #         'mse_loss': mse_loss,
        #         'L1_loss': l1_loss,
        #         }
        
        return loss, {
            'mse_loss': mse_loss,
            'L1_loss': l1_loss,
            }


@criterion_register('simple_unet2d')
class SimpleUnetCriterion(SimpleCriterion):
    pass


@criterion_register('simple_unet3d')
class SimpleUnetCriterion(SimpleCriterion):
    pass