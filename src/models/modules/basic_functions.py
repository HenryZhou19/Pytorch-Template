import warnings

import numpy as np
import torch
from torch.autograd import Function


class DifferentiableBinarization(Function):
    """
    __class__.apply(x) -> y
        x: [N, *] 0. ~ 1.
        y: [N, *] bool
    """
    def forward(self, x, threshold=None):
        if threshold is None:
            threshold = torch.mean(x, dim=tuple(range(1, x.dim())), keepdim=True)
        self.save_for_backward(x)
        y = torch.greater_equal(x, threshold)
        return y
    
    def backward(self, y_grad):
        x, = self.saved_tensors
        x_grad = y_grad * torch.where(torch.bitwise_and(x > 0., x < 1.), 1., 0.)
        # TODO: x_grad = y_grad  # Is the operation above necessary? Can it be replaced by this line?
        return x_grad
    
    
def masked_mean_and_var(inputs: torch.Tensor, mask_keep: torch.Tensor, dim=None, keepdim=False, need_var=True, unbiased_var=False):
    """
    inputs: Tensor of shape (*)
    mask_keep: Tensor of shape (*) type: bool
    """
    assert inputs.shape == mask_keep.shape
    if dim is None:
        dim = tuple(range(mask_keep.dim()))
        
    restored_dim = list(inputs.shape)
    if isinstance(dim, int):
        restored_dim[dim] = 1
    else:
        assert isinstance(dim, tuple)
        for d in dim:
            restored_dim[d] = 1
    restored_dim = tuple(restored_dim)
    
    count_nonzero = torch.count_nonzero(mask_keep, dim=dim)
    if torch.any(count_nonzero == 0).item():
        warnings.warn('No nonzero element in mask_keep')
    mean = torch.sum(inputs * mask_keep, dim=dim) / count_nonzero

    if need_var:
        if unbiased_var:
            variance = torch.sum((inputs - mean.reshape(restored_dim)) ** 2 * mask_keep, dim=dim) / (count_nonzero - 1)
        else:
            variance = torch.sum((inputs - mean.reshape(restored_dim)) ** 2 * mask_keep, dim=dim) / count_nonzero
        if keepdim:
            mean = mean.reshape(restored_dim)
            variance = variance.reshape(restored_dim)
        return mean, variance
    else:
        if keepdim:
            mean = mean.reshape(restored_dim)
        return mean


def numpy_get_local_maxima_with_topk(x: np.ndarray, k: int, lower_bound: float):
    x_extended = np.concatenate([np.array([lower_bound]), x, np.array([lower_bound])])
    local_maxima_indices = np.where((x_extended[1:-1] >= x_extended[:-2]) & (x_extended[1:-1] >= x_extended[2:]) & (x_extended[1:-1] > lower_bound))[0]

    if len(local_maxima_indices) >= k:
        local_maxima_values = x[local_maxima_indices]
        sorted_indices = np.argsort(local_maxima_values)
        k_max_indices = local_maxima_indices[sorted_indices[-k:]]
    else:
        n = len(local_maxima_indices)
        remaining_indices = np.setdiff1d(np.arange(len(x)), local_maxima_indices)
        sorted_remaining_indices = np.argsort(x[remaining_indices])
        k_max_indices = np.concatenate((local_maxima_indices, remaining_indices[sorted_remaining_indices[-(k - n):]]))

    return np.sort(k_max_indices)


def torch_get_local_maxima_with_topk(x: torch.Tensor, k: int, lower_bound: float):
    x_extended = torch.concatenate([torch.tensor([lower_bound], device=x.device), x, torch.tensor([lower_bound], device=x.device)])
    local_maxima_indices = torch.nonzero((x_extended[1:-1] >= x_extended[:-2]) & (x_extended[1:-1] >= x_extended[2:]) & (x_extended[1:-1] > lower_bound)).reshape(-1)
    
    if len(local_maxima_indices) >= k:
        local_maxima_values = x[local_maxima_indices]
        top_k_indices = torch.topk(local_maxima_values, k)[1]
        k_max_indices = local_maxima_indices[top_k_indices]
    else:
        n = len(local_maxima_indices)
        remaining_indices = torch.tensor([i for i in range(len(x)) if i not in local_maxima_indices], device=x.device)
        top_k_indices = torch.topk(x[remaining_indices], k - n)[1]
        k_max_indices = torch.cat((local_maxima_indices, remaining_indices[top_k_indices]))

    return torch.sort(k_max_indices)[0]


def create_group_attn_mask(total_length, group_length, batch_size=None):
    assert total_length % group_length == 0
    group_num = total_length // group_length
    
    attn_mask = torch.eye(group_num).bool()
    attn_mask = attn_mask.repeat(1, group_length).reshape(total_length, group_num).unsqueeze(-1)
    attn_mask = attn_mask.repeat(1, 1, 1, group_length).reshape(total_length, total_length)
    attn_mask: torch.Tensor = ~attn_mask
    
    if batch_size is None:
        return attn_mask
    else:
        return attn_mask.unsqueeze(0).expand(batch_size, *attn_mask.shape)
