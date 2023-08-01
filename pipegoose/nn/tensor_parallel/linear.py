from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from pipegoose.distributed.context import ParallelContext
from pipegoose.distributed.mode import ParallelMode


class ParallelLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, parallel_context: Optional[ParallelContext] = None
    ):
        super().__init__()
        local_world_size = parallel_context.get_local_world_size(ParallelMode.TENSOR)
        out_per_partition = self._get_output_per_partition(out_features, local_world_size)

        self.parallel_context = parallel_context
        self.weight = nn.Parameter(torch.randn(out_per_partition, in_features))

        if bias is True:
            self.bias = nn.Parameter(torch.randn(out_per_partition))

    def _get_output_per_partition(self, output_size: int, local_world_size: int) -> int:
        return output_size // local_world_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.linear(input, self.weight, self.bias)
        return output
