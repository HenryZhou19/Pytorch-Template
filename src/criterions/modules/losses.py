from typing import List, Union

import torch
import torch.nn as nn


def reduce_loss(loss: torch.Tensor, reduction):
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()

def one_hot_after_batch(x: torch.Tensor):
    x = torch.nn.functional.one_hot(x)
    new_dim_order = list(range(x.dim()))
    new_dim_order.insert(1, new_dim_order.pop())
    x = x.permute(new_dim_order)
    return x


class DiceLoss(nn.Module):
    """
    outputs: Tensor of shape (batch_size, ...) type: float, output score of foreground
    targets: Tensor of shape (batch_size, ...) type: int, ground truth class(foreground or background)
    """

    def __init__(self, smooth=1e-8, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, outputs, targets):
        batch_size = targets.size(0)

        outputs = outputs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)

        intersection = (outputs * targets).sum(1)
        union = outputs.sum(1) + targets.sum(1)

        loss = 1 - 2 * (intersection + self.smooth) / (union + self.smooth)

        return reduce_loss(loss, self.reduction)


class MulticlassDiceLoss(nn.Module):
    """
    outputs: Tensor of shape (batch_size, classes, ...) type: float, output scores of classes
    targets: Tensor of shape (batch_size, ...) type: int, ground truth class
    """

    def __init__(self, classes, weights=None, reduction='mean'):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.classes = classes
        self.weights = weights if weights is not None else torch.ones(classes)  # uniform weights for all classes
        self.reduction = reduction

    def forward(self, outputs, targets):
        assert self.classes == outputs.shape[1], f'MulticlassDiceLoss: classes {self.classes} does not match targets shape {targets.shape}'
        targets = one_hot_after_batch(targets)

        loss = 0
        for c in range(self.classes):
            loss += self.dice_loss(outputs[:, c], targets[:, c]) * self.weights[c]

        return reduce_loss(loss, self.reduction)


class FocalLoss(nn.Module):
    """
    outputs: Tensor of shape (batch_size, ...) type: float, output score of foreground
    targets: Tensor of shape (batch_size, ...) type: int, ground truth class(foreground or background)
    """

    def __init__(self, alpha=0.25, gamma=2, weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # just as [0.75, 0.25] in MulticlassFocalLoss
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight, reduction='none')  # sigmoid + BCEloss

    def forward(self, outputs, targets):
        batch_size = targets.size(0)

        log_p_t = -self.bce_fn(outputs, targets.float())
        p_t = torch.exp(log_p_t)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = -alpha_t * (1 - p_t) ** self.gamma * log_p_t
        loss = loss.view(batch_size, -1).mean(1)

        return reduce_loss(loss, self.reduction)


class MulticlassFocalLoss(nn.Module):
    """
    outputs: Tensor of shape (batch_size, classes, ...) type: float, output scores of classes
    targets: Tensor of shape (batch_size, ...) type: int, indicates the ground truth class
    """
    def __init__(self, classes, alpha: Union[float, List[float]] = 0.25, gamma=2, weight=None, ignore_index=-100, reduction='mean', device='cuda'):
        super().__init__()
        self.classes = classes
        if type(alpha) == list:
            assert len(alpha) == classes
        else:
            alpha = [alpha] * classes
        self.alpha = torch.as_tensor(self.alpha, device=device)
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, reduction='none',
                                         ignore_index=self.ignore_index)  # raw scores in

    def forward(self, outputs, targets):
        assert self.classes == outputs.shape[
            1], f'MulticlassDiceLoss: classes {self.classes} does not match targets shape {targets.shape}'
        batch_size = targets.size(0)
        alpha = self.alpha[targets]

        log_p_t = -self.ce_fn(outputs, targets)
        p_t = torch.exp(log_p_t)
        loss = -alpha * (1 - p_t) ** self.gamma * log_p_t

        loss = loss.view(batch_size, -1).mean(1)
        return reduce_loss(loss, self.reduction), log_p_t
    