from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.tensor_parallel._functional import (
    broadcast_to_tensor_group,
    gather_to_tensor_group,
    reduce_to_tensor_group,
    scatter_to_tensor_group,
)


class ColumnParallelLinear(nn.Module):
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

        self.gather_output = gather_output
        self.parallel_context = parallel_context
        self.weight = nn.Parameter(torch.randn(out_per_partition, in_features))

        if bias is True:
            self.bias = nn.Parameter(torch.randn(out_per_partition))
        else:
            self.bias = None
    
    def _get_output_per_partition(self, out_features: int, parallel_context: ParallelContext) -> int:
        local_world_size = parallel_context.get_world_size(ParallelMode.TENSOR)
        return out_features // local_world_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_parallel = broadcast_to_tensor_group(input, self.parallel_context)
        outputs = F.linear(input_parallel, self.weight, self.bias)

        if self.gather_output:
            outputs = gather_to_tensor_group(outputs, dim=-1, parallel_context=self.parallel_context)

        return outputs


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        parallel_context: Optional[ParallelContext] = None,
    ) -> None:
        super().__init__()
        in_per_partition = self._get_input_per_partition(in_features, parallel_context)

        self.parallel_context = parallel_context
        self.weight = nn.Parameter(torch.randn(out_features, in_per_partition))

        if bias is True:
            self.bias = nn.Parameter(torch.randn(out_features))

    def _get_input_per_partition(self, in_features: int, parallel_context: ParallelContext) -> int:
        local_world_size = parallel_context.get_world_size(ParallelMode.TENSOR)
        return in_features // local_world_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_parallel = scatter_to_tensor_group(input, dim=-1, parallel_context=self.parallel_context)
        output_parallel = F.linear(input_parallel, self.weight)
        outputs = reduce_to_tensor_group(output_parallel, parallel_context=self.parallel_context)

        if self.bias is not None:
            outputs = outputs + self.bias

        return outputs
