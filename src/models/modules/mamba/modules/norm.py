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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight
    
    
class RMSNormGated(torch.nn.Module):

    def __init__(self, normalized_shape, eps: float = 1e-5, norm_before_gate=True, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        self.register_parameter("bias", None)
        self.reset_parameters()
        self.norm_before_gate = norm_before_gate

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor, z=None) -> torch.Tensor:
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))
        """
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight
        
        
    def forward(self, x, z=None):
        if self.norm_before_gate:
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
            if z is not None:
                x = x * silu(z)
        else:
            if z is not None:
                x = x * silu(z)
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return


def silu(x):
    return x * torch.sigmoid(x)