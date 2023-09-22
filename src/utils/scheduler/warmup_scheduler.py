import math
from bisect import bisect_right
from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class WarmUpCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, T_warmup, warmup_factor, lr_min, last_epoch=-1):
        assert T_max > T_warmup, "T_max should be larger than T_warmup."
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.warmup_factor = warmup_factor
        self.lr_min = lr_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            alpha = float(self.last_epoch) / self.T_warmup
            alpha = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            alpha = float(self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup)
            alpha = 0.5 + 0.5 * math.cos(math.pi * alpha)
            return [self.lr_min + alpha * (base_lr - self.lr_min) for base_lr in self.base_lrs]


class WarmupLinearLR(_LRScheduler):
    def __init__(self, optimizer, T_max, T_warmup, warmup_factor, lr_min, last_epoch=-1):
        assert T_max > T_warmup, "T_max should be larger than T_warmup."
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.warmup_factor = warmup_factor
        self.lr_min = lr_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            alpha = float(self.last_epoch) / self.T_warmup
            alpha = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            alpha = float(self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup)
            alpha = 1 - alpha
            return [self.lr_min + alpha * (base_lr - self.lr_min) for base_lr in self.base_lrs]


class WarmupMultiStepLR(_LRScheduler):
    def __init__(self, optimizer, step_milestones: List[int], gamma, T_max, T_warmup, warmup_factor, lr_min, last_epoch=-1):
        assert list(step_milestones) == sorted(step_milestones), "MultiStepLR milestones should be a list of increasing integers."
        assert T_max > step_milestones[-1], "T_max should be larger than the last milestone."
        assert T_warmup < step_milestones[0], "T_warmup should be smaller than the first milestone."
        self.milestones = step_milestones
        self.gamma = gamma
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.warmup_factor = warmup_factor
        self.lr_min = lr_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            alpha = float(self.last_epoch) / self.T_warmup
            alpha = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            alpha = self.gamma ** bisect_right(self.milestones, self.last_epoch)
            return [self.lr_min + alpha * (base_lr - self.lr_min) for base_lr in self.base_lrs]

