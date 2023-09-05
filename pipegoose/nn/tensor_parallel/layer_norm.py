import torch
import torch.nn.functional as F
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5, bias: bool = True, parallel_context: ParallelContext = None):
        super().__init__()
        assert parallel_context is not None, "parallel_context must be provided"

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.parallel_context = parallel_context

        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normalized_shape = (self.normalized_shape,) if isinstance(self.normalized_shape, int) else self.normalized_shape
        return F.layer_norm(input, normalized_shape, self.weight, self.bias, self.eps)
