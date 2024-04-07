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
    ]


class WarmUpFn:
    constant = lambda last_epoch, T_warmup: 0.0
    linear = lambda last_epoch, T_warmup: float(last_epoch) / T_warmup
    exponential = lambda last_epoch, T_warmup, gamma=5.0: math.exp(gamma * float(last_epoch) / T_warmup - gamma)
    cosine = lambda last_epoch, T_warmup: 0.5 * (1.0 - math.cos(math.pi * float(last_epoch) / T_warmup))
    
    def get_warmup_fn(warmup_type, warmup_factor):
        if warmup_type == 'constant' and warmup_factor < 1e-8:
            Warning(f'warmup_factor = {warmup_factor} is too small for constant warmup.')
        return getattr(WarmUpFn, warmup_type)


class _AmpStepLR(_LRScheduler):  # remove the 'call of `lr_scheduler.step()` before `optimizer.step()`' warning when use amp or grad_accumulation
    @staticmethod
    def with_counter(method, is_scaler_step=False):
        instance_ref = weakref.ref(method.__self__)
        func = method.__func__
        cls = instance_ref().__class__
        del method
        @wraps(func)
        def wrapper(*args, **kwargs):
            if is_scaler_step:
                optimizer_in_scaler_call = args[0]
                optimizer_in_scaler_call._step_count += 1
            instance = instance_ref()
            wrapped = func.__get__(instance, cls)
            return wrapped(*args, **kwargs)
        wrapper._with_counter = True
        return wrapper
    
    def __init__(self, optimizer, scaler, do_grad_accumulation, last_epoch):
        if scaler is not None:  # prevent _LRScheduler to wrap optimizer.step()
            optimizer.step = self.with_counter(optimizer.step)  
        super().__init__(optimizer, last_epoch)
        if scaler is not None:  # wrap scaler.step() to replace the number of optimizer.step() calls
            scaler.step = self.with_counter(scaler.step, is_scaler_step=True)
        if do_grad_accumulation:  # just avoid the warning when use grad_accumulation
            optimizer._step_count = 1
            
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer' and key != 'warmup_fn'}


class WarmUpVanillaLR(_AmpStepLR):
    def __init__(self, optimizer, scaler, do_grad_accumulation, T_max, T_warmup, warmup_factor, warmup_fn, last_epoch=-1):
        assert T_max > T_warmup, 'T_max should be larger than T_warmup.'
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.warmup_factor = warmup_factor
        self.warmup_fn = warmup_fn
        super().__init__(optimizer, scaler, do_grad_accumulation, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            alpha = self.warmup_fn(self.last_epoch, self.T_warmup)
            alpha = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            return [base_lr for base_lr in self.base_lrs]


class WarmUpCosineAnnealingLR(_AmpStepLR):
    def __init__(self, optimizer, scaler, do_grad_accumulation, T_max, T_warmup, warmup_factor, lr_min, warmup_fn, last_epoch=-1):
        assert T_max > T_warmup, 'T_max should be larger than T_warmup.'
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.warmup_factor = warmup_factor
        self.warmup_fn = warmup_fn
        self.lr_min = lr_min
        super().__init__(optimizer, scaler, do_grad_accumulation, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            alpha = self.warmup_fn(self.last_epoch, self.T_warmup)
            alpha = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            alpha = float(self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup)
            alpha = 0.5 + 0.5 * math.cos(math.pi * alpha)
            return [self.lr_min + alpha * (base_lr - self.lr_min) for base_lr in self.base_lrs]


class WarmUpLinearLR(_AmpStepLR):
    def __init__(self, optimizer, scaler, do_grad_accumulation, T_max, T_warmup, warmup_factor, lr_min, warmup_fn, last_epoch=-1):
        assert T_max > T_warmup, 'T_max should be larger than T_warmup.'
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.warmup_factor = warmup_factor
        self.warmup_fn = warmup_fn
        self.lr_min = lr_min
        super().__init__(optimizer, scaler, do_grad_accumulation, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            alpha = self.warmup_fn(self.last_epoch, self.T_warmup)
            alpha = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            alpha = float(self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup)
            alpha = 1 - alpha
            return [self.lr_min + alpha * (base_lr - self.lr_min) for base_lr in self.base_lrs]


class WarmUpMultiStepLR(_AmpStepLR):
    def __init__(self, optimizer, scaler, do_grad_accumulation, step_milestones: List[int], gamma, T_max, T_warmup, warmup_factor, lr_min, warmup_fn, last_epoch=-1):
        assert list(step_milestones) == sorted(step_milestones), 'MultiStepLR milestones should be a list of increasing integers.'
        assert T_max > step_milestones[-1], 'T_max should be larger than the last milestone.'
        assert T_warmup < step_milestones[0], 'T_warmup should be smaller than the first milestone.'
        self.milestones = step_milestones
        self.gamma = gamma
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.warmup_factor = warmup_factor
        self.warmup_fn = warmup_fn
        self.lr_min = lr_min
        super().__init__(optimizer, scaler, do_grad_accumulation, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            alpha = self.warmup_fn(self.last_epoch, self.T_warmup)
            alpha = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            alpha = self.gamma ** bisect_right(self.milestones, self.last_epoch)
            return [self.lr_min + alpha * (base_lr - self.lr_min) for base_lr in self.base_lrs]

