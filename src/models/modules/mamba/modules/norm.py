import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-5, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        var = input.pow(2).mean(dim=-1, keepdim=True) + self.eps
        input_norm = input * torch.rsqrt(var)
        rmsnorm = self.weight * input_norm
        return rmsnorm