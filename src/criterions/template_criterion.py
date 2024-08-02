import torch
from torch import nn

from .modules.criterion_base import CriterionBase, criterion_register


@criterion_register('simple')
class SimpleCriterion(CriterionBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def _get_iter_loss_and_metrics(self, outputs, targets):
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

        # if self.infer_mode:
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


@criterion_register('lenet')
class MnistCriterion(CriterionBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ce_loss = nn.CrossEntropyLoss()
        self.epoch_sample_count = 0
        self.epoch_correct_count = 0
        
    def _get_iter_loss_and_metrics(self, outputs, targets):
        pred_scores = outputs['pred_scores']
        gt_y = targets['gt_y']   

        # metrics (loss) used for backprop
        if self.loss_config == 'ce':
            ce_loss = self.ce_loss(pred_scores, gt_y)
            loss = 1 * ce_loss
        else:
            raise NotImplementedError(f'loss "{self.loss_config}" has not been implemented yet.')
        
        
        _, predicted = torch.max(pred_scores.data, 1)
        self.epoch_sample_count += gt_y.shape[0]
        self.epoch_correct_count += (predicted == gt_y).sum().item()
            
        return loss, {
            'ce_loss': ce_loss,
            }

    def _get_epoch_metrics_and_reset(self):
        import torch.distributed as dist

        from src.utils.misc import DistMisc

        if DistMisc.is_dist_avail_and_initialized():
            self.epoch_sample_count = torch.tensor(self.epoch_sample_count).cuda()
            self.epoch_correct_count = torch.tensor(self.epoch_correct_count).cuda()
            dist.all_reduce(self.epoch_sample_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.epoch_correct_count, op=dist.ReduceOp.SUM)
        accuracy = self.epoch_correct_count / self.epoch_sample_count
        self.epoch_correct_count = 0
        self.epoch_sample_count = 0
        return {
            'accuracy': accuracy,
            }