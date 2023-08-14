from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.tensor_parallel._operations import (
    broadcast_tensor_1d,
    gather_tensor_1d,
)


class ParallelColumnLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
        parallel_context: Optional[ParallelContext] = None,
    ):
        super().__init__()
        out_per_partition = self._get_output_per_partition(out_features, parallel_context)

        self.parallel_context = parallel_context
        self.gather_output = gather_output
        self.weight = nn.Parameter(torch.randn(out_per_partition, in_features))

        if bias is True:
            self.bias = nn.Parameter(torch.randn(out_per_partition))

    def _get_output_per_partition(self, out_features: int, parallel_context: ParallelContext) -> int:
        local_world_size = parallel_context.get_world_size(ParallelMode.TENSOR)
        return out_features // local_world_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_parallel = broadcast_tensor_1d(input, self.parallel_context)
        outputs = F.linear(input_parallel, self.weight, self.bias)

        # if self.bias is True:
        #     outputs = outputs + self.bias

        if self.gather_output:
            outputs = gather_tensor_1d(outputs, dim=-1, parallel_context=self.parallel_context)

        return outputs
