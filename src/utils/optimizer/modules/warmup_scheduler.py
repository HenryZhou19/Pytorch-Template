import math
import weakref
from bisect import bisect_right
from functools import wraps
from typing import List

from torch.optim.lr_scheduler import _LRScheduler

__all__ = [
    'WarmUpFn',
    'WarmUpVanillaLR',
    'WarmUpCosineAnnealingLR',
    'WarmUpLinearLR',
    'WarmUpMultiStepLR',
    'WarmupCosineAnnealingRestartLR',
    'WarmupCosineAnnealingMultiCycleLR',
    ]


class WarmUpFn:
    no_warmup = lambda last_epoch, T_warmup: 1.0
    constant = lambda last_epoch, T_warmup: 0.0
    linear = lambda last_epoch, T_warmup: float(last_epoch) / T_warmup
    exponential = lambda last_epoch, T_warmup, gamma=5.0: math.exp(gamma * float(last_epoch) / T_warmup - gamma)
    cosine = lambda last_epoch, T_warmup: 0.5 * (1.0 - math.cos(math.pi * float(last_epoch) / T_warmup))
    
    def get_warmup_fn(warmup_type):
        return getattr(WarmUpFn, warmup_type)


class _CustomedStepLR(_LRScheduler):  # remove the 'call of `lr_scheduler.step()` before `optimizer.step()`' warning when grad_accumulation
    def __init__(self, optimizer, scaler, do_grad_accumulation, last_epoch): 
        super().__init__(optimizer, last_epoch)
        if do_grad_accumulation:  # just avoid the warning when use grad_accumulation
            setattr(optimizer, '_opt_called', True)
            
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer' and key != 'warmup_fn'}


class WarmUpVanillaLR(_CustomedStepLR):
    def __init__(self, optimizer, scaler, do_grad_accumulation, T_max, T_warmup, lr_min_factor, warmup_fn, last_epoch=-1):
        assert T_max > T_warmup, 'T_max should be larger than T_warmup.'
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.warmup_fn = warmup_fn
        self.lr_min_factor = lr_min_factor
        self.min_lrs = [self.lr_min_factor * group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, scaler, do_grad_accumulation, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            alpha = self.warmup_fn(self.last_epoch, self.T_warmup)
        else:
            alpha = 1.0
        return [min_lr + alpha * (base_lr - min_lr) for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)]


class WarmUpCosineAnnealingLR(_CustomedStepLR):
    def __init__(self, optimizer, scaler, do_grad_accumulation, T_max, T_warmup, lr_min_factor, warmup_fn, last_epoch=-1):
        assert T_max > T_warmup, 'T_max should be larger than T_warmup.'
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.warmup_fn = warmup_fn
        self.lr_min_factor = lr_min_factor
        self.min_lrs = [self.lr_min_factor * group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, scaler, do_grad_accumulation, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            alpha = self.warmup_fn(self.last_epoch, self.T_warmup)
        else:
            alpha = float(self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup)
            alpha = 0.5 + 0.5 * math.cos(math.pi * alpha)
        return [min_lr + alpha * (base_lr - min_lr) for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)]


class WarmUpLinearLR(_CustomedStepLR):
    def __init__(self, optimizer, scaler, do_grad_accumulation, T_max, T_warmup, lr_min_factor, warmup_fn, last_epoch=-1):
        assert T_max > T_warmup, 'T_max should be larger than T_warmup.'
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.warmup_fn = warmup_fn
        self.lr_min_factor = lr_min_factor
        self.min_lrs = [self.lr_min_factor * group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, scaler, do_grad_accumulation, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            alpha = self.warmup_fn(self.last_epoch, self.T_warmup)
        else:
            alpha = float(self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup)
            alpha = 1 - alpha
        return [min_lr + alpha * (base_lr - min_lr) for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)]


class WarmUpMultiStepLR(_CustomedStepLR):
    def __init__(self, optimizer, scaler, do_grad_accumulation, step_milestones: List[int], gamma, T_max, T_warmup, lr_min_factor, warmup_fn, last_epoch=-1):
        assert list(step_milestones) == sorted(step_milestones), 'MultiStepLR milestones should be a list of increasing integers.'
        assert T_max > step_milestones[-1], 'T_max should be larger than the last milestone.'
        assert T_warmup < step_milestones[0], 'T_warmup should be smaller than the first milestone.'
        self.milestones = step_milestones
        self.gamma = gamma
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.warmup_fn = warmup_fn
        self.lr_min_factor = lr_min_factor
        self.min_lrs = [self.lr_min_factor * group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, scaler, do_grad_accumulation, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            alpha = self.warmup_fn(self.last_epoch, self.T_warmup)
        else:
            alpha = self.gamma ** bisect_right(self.milestones, self.last_epoch)
        return [min_lr + alpha * (base_lr - min_lr) for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)]


class WarmupCosineAnnealingRestartLR(_CustomedStepLR):
    """
    Hacked from https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py

        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: 1.0
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self, optimizer, scaler, do_grad_accumulation, T_warmup, lr_min_factor, warmup_fn,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert T_warmup < first_cycle_steps, 'T_warmup should be smaller than first_cycle_steps.'
        assert cycle_mult >= 1.0, 'cycle_mult should be greater than or equal to 1.'
        
        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.T_warmup = T_warmup
        self.warmup_fn = warmup_fn
        self.gamma = gamma  # decrease rate of max learning rate by cycle
        self.lr_min_factor = lr_min_factor
        self.min_lrs = [self.lr_min_factor * group['lr'] for group in optimizer.param_groups]
        
        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super().__init__(optimizer, scaler, do_grad_accumulation, last_epoch)
        # get self.base_lrs, self.last_epoch and do self.step() once

    def get_lr(self):
        if self.step_in_cycle < self.T_warmup:
            alpha = self.warmup_fn(self.step_in_cycle, self.T_warmup)
            cycle_min_lrs = [cycle_min_lr / self.gamma for cycle_min_lr in self.cycle_min_lrs]
        else:
            alpha = float(self.step_in_cycle - self.T_warmup) / (self.cur_cycle_steps - self.T_warmup)
            alpha = 0.5 + 0.5 * math.cos(math.pi * alpha)
            cycle_min_lrs = self.cycle_min_lrs
        return [alpha * (cycle_max_lr - min_lr) + min_lr for cycle_max_lr, min_lr in zip(self.cycle_max_lrs, cycle_min_lrs)]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(self.cur_cycle_steps * self.cycle_mult)
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (
                                epoch / self.first_cycle_steps * (self.cycle_mult - 1)
                                + 1
                            ),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult ** n - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (
                        n
                    )
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.cycle_max_lrs = [base_lr * (self.gamma ** self.cycle) for base_lr in self.base_lrs]
        self.cycle_min_lrs = [self.lr_min_factor * cycle_max_lr for cycle_max_lr in self.cycle_max_lrs]
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        
        
class WarmupCosineAnnealingMultiCycleLR(_CustomedStepLR):
    """
    Hacked from https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py

        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: 1.0
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self, optimizer, scaler, do_grad_accumulation, T_warmup, lr_min_factor, warmup_fn,
        cycle_steps_list: List[int],
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        for cycle_steps in cycle_steps_list:
            assert T_warmup < cycle_steps, 'T_warmup should be smaller than cycle_steps.'

        self.cycle_steps_list = cycle_steps_list
        self.T_warmup = T_warmup
        self.warmup_fn = warmup_fn
        self.gamma = gamma  # decrease rate of max learning rate by cycle
        self.lr_min_factor = lr_min_factor
        self.min_lrs = [self.lr_min_factor * group['lr'] for group in optimizer.param_groups]
        
        self.cycle = 0  # cycle count
        self.cycle_type = 0  # cycle type count
        self.step_in_cycle = last_epoch  # step size of the current cycle
        self.cur_cycle_steps = self.cycle_steps_list[0]  # first cycle step size

        super().__init__(optimizer, scaler, do_grad_accumulation, last_epoch)
        # get self.base_lrs, self.last_epoch and do self.step() once
        
    def get_lr(self):
        if self.step_in_cycle < self.T_warmup:
            alpha = self.warmup_fn(self.step_in_cycle, self.T_warmup)
            cycle_min_lrs = [cycle_min_lr / self.gamma for cycle_min_lr in self.cycle_min_lrs]
        else:
            alpha = float(self.step_in_cycle - self.T_warmup) / (self.cur_cycle_steps - self.T_warmup)
            alpha = 0.5 + 0.5 * math.cos(math.pi * alpha)
            cycle_min_lrs = self.cycle_min_lrs
        return [alpha * (cycle_max_lr - min_lr) + min_lr for cycle_max_lr, min_lr in zip(self.cycle_max_lrs, cycle_min_lrs)]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.cycle_type = self.cycle % len(self.cycle_steps_list)
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = self.cur_cycle_steps = self.cycle_steps_list[self.cycle_type]  # current cycle step size
        else:
            if epoch >= self.cycle_steps_list[0]:
                full_cycle_groups = epoch // sum(self.cycle_steps_list)
                remain_steps = epoch % sum(self.cycle_steps_list)
                n = full_cycle_groups * len(self.cycle_steps_list)
                for cycle_steps in self.cycle_steps_list:
                    if remain_steps < cycle_steps:
                        self.cur_cycle_steps = cycle_steps
                        self.step_in_cycle = remain_steps
                        break
                    remain_steps -= cycle_steps
                    n += 1
                self.cycle = n
            else:
                self.cur_cycle_steps = self.cycle_steps_list[0]
                self.step_in_cycle = epoch

        self.cycle_max_lrs = [base_lr * (self.gamma ** (self.cycle // len(self.cycle_steps_list))) for base_lr in self.base_lrs]
        self.cycle_min_lrs = [self.lr_min_factor * cycle_max_lr for cycle_max_lr in self.cycle_max_lrs]
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]