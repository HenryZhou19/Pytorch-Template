import warnings
from typing import Union

import numpy as np
import torch
from torch import nn
from torch.autograd import Function


def trunc_normal_init_linear_weights(module: torch.nn.Module, std: float = 0.02):
    """
    Initialize the weights of a linear layer with truncated normal distribution.
    
    Args:
        module (torch.nn.Linear): The linear layer to initialize.
        std (float): The standard deviation of the truncated normal distribution.
    """
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


class DifferentiableBinarization(Function):
    """
    __class__.apply(x) -> y
        x: [N, *] 0. ~ 1.
        y: [N, *] bool
    """
    @staticmethod
    def forward(ctx, x, threshold=None):
        if threshold is None:
            threshold = torch.mean(x, dim=tuple(range(1, x.dim())), keepdim=True)
        ctx.save_for_backward(x)
        y = torch.greater_equal(x, threshold)
        return y
    
    @staticmethod
    def backward(ctx, y_grad):
        x, = ctx.saved_tensors
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


def create_group_attn_mask(total_length: int, group_length: int, batch_size: int=None):
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


def multi_dim_repeat_interleave(tensor: torch.Tensor, repeats: Union[list, tuple]) -> torch.Tensor:
    """
    repeat_interleave on multiple dimensions.
    
    Args:
        tensor (torch.Tensor): input tensor [*].
        repeats (list or tuple): the number of repetitions for each dimension.
            if some elements in `repeats` are `1`, the corresponding dimensions will be skipped.

    Returns:
        torch.Tensor: output tensor [*].
    """
    assert isinstance(repeats, (list, tuple)), "`repeats` must be a list or tuple."
    assert len(repeats) == tensor.dim(), "The length of `repeats` must be equal to the number of dimensions of `tensor`."
    
    for dim, repeat in enumerate(repeats):
        if repeat == 1:
            continue
        tensor = tensor.repeat_interleave(repeat, dim=dim)
    return tensor


def adapt_conv_2d_load_from_state_dict(state_dict, module_prefix, current_module, strict=False):
    weight_key = module_prefix + 'weight'
    current_weight = current_module.weight
    if weight_key in state_dict:
        loaded_weight = state_dict[weight_key]
        c_out, c_in, kh, kw = current_weight.shape
        
        try:
            c_out_l, c_in_l, kh_l, kw_l = loaded_weight.shape
            adapted = False
            with torch.no_grad():
                if c_out_l != c_out:
                    if c_out_l > c_out:
                        loaded_weight = loaded_weight[:c_out]
                    else:
                        pad_shape = (c_out - c_out_l, c_in_l, kh_l, kw_l)
                        loaded_weight = torch.cat([loaded_weight, loaded_weight.new_zeros(pad_shape)], dim=0)
                    adapted = True
                
                if c_in != c_in_l:
                    if c_in_l == 1 and c_in > 1:
                        loaded_weight = loaded_weight / c_in
                        loaded_weight = loaded_weight.repeat(1, c_in, 1, 1)
                    elif c_in == 1 and c_in_l > 1:
                        loaded_weight = loaded_weight.mean(dim=1, keepdim=True)
                    elif c_in > c_in_l:
                        pad_shape = (loaded_weight.shape[0], c_in - c_in_l, loaded_weight.shape[2], loaded_weight.shape[3])
                        loaded_weight = torch.cat([loaded_weight, loaded_weight.new_zeros(pad_shape)], dim=1)
                    else:  # c_in < c_in_l
                        loaded_weight = loaded_weight[:, :c_in, ...]
                    adapted = True
                
                if (kh, kw) != (kh_l, kw_l):
                    loaded_weight = nn.functional.interpolate(loaded_weight, size=(kh, kw), mode='bilinear', align_corners=False)
                    adapted = True
                
                if adapted:
                    loaded_weight = loaded_weight.to(dtype=current_weight.dtype, device=current_weight.device)
                    print(f"Adapted conv2d weights for key '{weight_key}' from shape {loaded_weight.shape} to {current_weight.shape}.")
        except:
            if strict:
                raise ValueError(f"Failed to adapt conv2d weights for key '{weight_key}'. The loaded shape {loaded_weight.shape} cannot be adapted to the current shape {current_weight.shape}.")
            else:
                warnings.warn(f"Failed to adapt conv2d weights for key '{weight_key}'. The loaded shape {loaded_weight.shape} cannot be adapted to the current shape {current_weight.shape}. The weights will not be loaded.")
                loaded_weight = current_weight

        state_dict[weight_key] = loaded_weight
    
    if current_module.bias is not None:
        bias_key = module_prefix + 'bias'
        current_bias = current_module.bias
        if bias_key in state_dict:
            loaded_bias = state_dict[bias_key]
            c_out = current_bias.shape
            
            try:
                c_out_l = loaded_bias.shape
                with torch.no_grad():
                    if c_out_l != c_out:
                        if c_out_l > c_out:
                            loaded_bias = loaded_bias[:c_out]
                        else:
                            pad_shape = (c_out - c_out_l,)
                            loaded_bias = torch.cat([loaded_bias, loaded_bias.new_zeros(pad_shape)], dim=0)
                            
                    loaded_bias = loaded_bias.to(dtype=current_bias.dtype, device=current_bias.device)
                    print(f"Adapted bias weights for key '{bias_key}' from shape {loaded_bias.shape} to {current_bias.shape}.")
            except:
                if strict:
                    raise ValueError(f"Failed to adapt bias weights for key '{bias_key}'. The loaded shape {loaded_bias.shape} cannot be adapted to the current shape {current_bias.shape}.")
                else:
                    warnings.warn(f"Failed to adapt bias weights for key '{bias_key}'. The loaded shape {loaded_bias.shape} cannot be adapted to the current shape {current_bias.shape}. The weights will not be loaded.")
                    loaded_bias = current_bias
        
            state_dict[bias_key] = loaded_bias
        
    return state_dict


def adapt_conv_3d_load_from_state_dict(state_dict, module_prefix, current_module, strict=False):
    weight_key = module_prefix + 'weight'
    current_weight = current_module.weight
    if weight_key in state_dict:
        loaded_weight = state_dict[weight_key]
        c_out, c_in, kd, kh, kw = current_weight.shape
        
        try:
            c_out_l, c_in_l, kd_l, kh_l, kw_l = loaded_weight.shape
            adapted = False
            with torch.no_grad():
                if c_out_l != c_out:
                    if c_out_l > c_out:
                        loaded_weight = loaded_weight[:c_out]
                    else:
                        pad_shape = (c_out - c_out_l, c_in_l, kd_l, kh_l, kw_l)
                        loaded_weight = torch.cat([loaded_weight, loaded_weight.new_zeros(pad_shape)], dim=0)
                    adapted = True
                
                if c_in != c_in_l:
                    if c_in_l == 1 and c_in > 1:
                        loaded_weight = loaded_weight / c_in
                        loaded_weight = loaded_weight.repeat(1, c_in, 1, 1, 1)
                    elif c_in == 1 and c_in_l > 1:
                        loaded_weight = loaded_weight.mean(dim=1, keepdim=True)
                    elif c_in > c_in_l:
                        pad_shape = (loaded_weight.shape[0], c_in - c_in_l, loaded_weight.shape[2], loaded_weight.shape[3], loaded_weight.shape[4])
                        loaded_weight = torch.cat([loaded_weight, loaded_weight.new_zeros(pad_shape)], dim=1)
                    else:  # c_in < c_in_l
                        loaded_weight = loaded_weight[:, :c_in, ...]
                    adapted = True
                
                if (kd, kh, kw) != (kd_l, kh_l, kw_l):
                    loaded_weight = nn.functional.interpolate(loaded_weight, size=(kd, kh, kw), mode='trilinear', align_corners=False)
                    adapted = True
                
                if adapted:
                    loaded_weight = loaded_weight.to(dtype=current_weight.dtype, device=current_weight.device)
                    print(f"Adapted conv3d weights for key '{weight_key}' from shape {loaded_weight.shape} to {current_weight.shape}.")
        except:
            if strict:
                raise ValueError(f"Failed to adapt conv3d weights for key '{weight_key}'. The loaded shape {loaded_weight.shape} cannot be adapted to the current shape {current_weight.shape}.")
            else:
                warnings.warn(f"Failed to adapt conv3d weights for key '{weight_key}'. The loaded shape {loaded_weight.shape} cannot be adapted to the current shape {current_weight.shape}. The weights will not be loaded.")
                loaded_weight = current_weight

        state_dict[weight_key] = loaded_weight
    
    if current_module.bias is not None:
        bias_key = module_prefix + 'bias'
        current_bias = current_module.bias
        if bias_key in state_dict:
            loaded_bias = state_dict[bias_key]
            c_out = current_bias.shape
            
            try:
                c_out_l = loaded_bias.shape
                with torch.no_grad():
                    if c_out_l != c_out:
                        if c_out_l > c_out:
                            loaded_bias = loaded_bias[:c_out]
                        else:
                            pad_shape = (c_out - c_out_l,)
                            loaded_bias = torch.cat([loaded_bias, loaded_bias.new_zeros(pad_shape)], dim=0)
                        
                        loaded_bias = loaded_bias.to(dtype=current_bias.dtype, device=current_bias.device)
                        print(f"Adapted bias weights for key '{bias_key}' from shape {loaded_bias.shape} to {current_bias.shape}.")
            except:
                if strict:
                    raise ValueError(f"Failed to adapt bias weights for key '{bias_key}'. The loaded shape {loaded_bias.shape} cannot be adapted to the current shape {current_bias.shape}.")
                else:
                    warnings.warn(f"Failed to adapt bias weights for key '{bias_key}'. The loaded shape {loaded_bias.shape} cannot be adapted to the current shape {current_bias.shape}. The weights will not be loaded.")
                    loaded_bias = current_bias
        
            state_dict[bias_key] = loaded_bias
        
    return state_dict


def adapt_L_C_parameter_load_from_state_dict(state_dict, param_key, current_param, strict=False):
    if param_key in state_dict:
        loaded_param = state_dict[param_key]
        L, C = current_param.shape
        
        try:
            L_l, C_l = loaded_param.shape
            with torch.no_grad():
                if L_l != L:
                    assert C_l == C, f"Loaded parameter shape {loaded_param.shape} does not match current parameter shape {current_param.shape}."
                    # Adjust the shape of the loaded parameter
                    loaded_param = loaded_param.permute(1, 0).reshape(1, C_l, L_l)
                    loaded_param = nn.functional.interpolate(loaded_param, size=(L,), mode='linear', align_corners=False)
                    loaded_param = loaded_param.reshape(C_l, L).permute(1, 0)
                
                    loaded_param = loaded_param.to(dtype=current_param.dtype, device=current_param.device)
                    print(f"Adapted parameter weights for key '{param_key}' from shape {loaded_param.shape} to {current_param.shape}.")
        except:
            if strict:
                raise ValueError(f"Failed to adapt parameter weights for key '{param_key}'. The loaded shape {loaded_param.shape} cannot be adapted to the current shape {current_param.shape}.")
            else:
                warnings.warn(f"Failed to adapt parameter weights for key '{param_key}'. The loaded shape {loaded_param.shape} cannot be adapted to the current shape {current_param.shape}. The weights will not be loaded.")
                loaded_param = current_param
        
        state_dict[param_key] = loaded_param
    
    return state_dict