import torch.distributed as dist

from pipegoose.distributed._initializers.initializer import (
    ProcessGroupInitializer,
    ProcessGroupResult,
)
from pipegoose.distributed.parallel_mode import ParallelMode


class TensorParallelGroupInitializer(ProcessGroupInitializer):
    def init_dist_group(self) -> ProcessGroupResult:
        num_tensor_parallel_groups = self.world_size // self.tensor_parallel_size
        local_rank = None
        process_group = None
        local_world_size = None
        ranks_in_group = None
        parallel_mode = ParallelMode.TENSOR

        for i in range(num_tensor_parallel_groups):
            ranks = list(range(i * self.tensor_parallel_size, (i + 1) * self.tensor_parallel_size))

            # NOTE: dist.new_group() must be called collectively by all the processes
            # that would be part of the group, which means every process in the group
            # needs to call this function. If only a subset of the processes call new_group(),
            # it will hang because it's waiting for the rest of the processes to join.
            group = dist.new_group(ranks=ranks)

            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                local_world_size = len(ranks)
                ranks_in_group = ranks
                process_group = group

        return {
            "local_rank": local_rank,
            "local_world_size": local_world_size,
            "ranks_in_group": ranks_in_group,
            "process_group": process_group,
            "parallel_mode": parallel_mode,
        }
