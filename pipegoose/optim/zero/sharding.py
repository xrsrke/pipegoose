import copy
from typing import Dict, List

import torch

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode


class ParameterSharding:
    """
    Shard optimizer parameters across parallelism dimension.

    NOTE: Only shard the parameters in each param groups and keep the number of param groups the same.
    """

    def __init__(
        self, param_groups: List[Dict[str, torch.Tensor]], parallel_context: ParallelContext, parallel_mode: ParallelMode
    ):
        self.param_groups = param_groups
        self.parallel_context = parallel_context
        self.parallel_mode = parallel_mode

    def shard(self) -> List[Dict[str, torch.Tensor]]:
        """
        Credit: https://github.com/facebookresearch/fairscale/blob/164cc0f3170b4a3951dd84dda29c3e1504ac4d6e/fairscale/optim/oss.py#L173
        """
        world_size = self.parallel_context.get_world_size(self.parallel_mode)
        partition_parameters = [[] for _ in range(world_size)]
        sizes = [0 for _ in range(world_size)]

        for param_group in self.param_groups:
            param_lists = [[] for _ in range(world_size)]

            for param in param_group["params"]:
                # TODO: fix if the numel of more than one ranks are equal
                next_rank = sizes.index(min(sizes))
                param_lists[next_rank].append(param)
                sizes[next_rank] += param.numel()

            for rank, params in enumerate(param_lists):
                partitioned_param_group = copy.copy(param_group)
                partitioned_param_group["params"] = params
                partition_parameters[rank].append(partitioned_param_group)

        return partition_parameters
