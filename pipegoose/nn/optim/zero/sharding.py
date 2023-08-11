import copy
from typing import Dict, List

import torch

from pipegoose.distributed.parallel_context import ParallelContext


class ParameterSharding:
    def __init__(self, param_groups: List[Dict[str, torch.Tensor]], parallel_context: ParallelContext):
        self.param_groups = param_groups
        self.parallel_context = parallel_context

    def shard(self) -> List[Dict[str, torch.Tensor]]:
        world_size = self.parallel_context.get_world_size()

        partition_parameters = [[] for _ in range(world_size)]
        sizes = [0 for _ in range(world_size)]

        for param_group in self.param_groups:
            param_lists = [[] for _ in range(world_size)]

            for param in param_group["params"]:
                next_rank = sizes.index(min(sizes))
                param_lists[next_rank].append(param)
                sizes[next_rank] += param.numel()

            for rank, params in enumerate(param_lists):
                partitioned_param_group = copy.copy(param_group)
                partitioned_param_group["params"] = params
                partition_parameters[rank].append(partitioned_param_group)

        return partition_parameters
