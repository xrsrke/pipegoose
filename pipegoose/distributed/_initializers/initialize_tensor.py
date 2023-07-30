import torch.distributed as dist

from pipegoose.distributed._initializers.initializer import (
    ProcessGroupInitializer,
    ProcessGroupResult,
)
from pipegoose.distributed.mode import ParallelMode


class TensorParallelGroupInitializer(ProcessGroupInitializer):
    def init_dist_group(self) -> ProcessGroupResult:
        backend = dist.get_backend()
        num_tensor_parallel_groups = self.world_size // self.tensor_parallel_size
        local_rank = None
        process_group = None
        local_world_size = None
        ranks_in_group = None
        parallel_mode = ParallelMode.TENSOR

        for i in range(num_tensor_parallel_groups):
            ranks = list(range(i * self.tensor_parallel_size, (i + 1) * self.tensor_parallel_size))

            if self.rank in ranks:
                process_group = dist.new_group(ranks=ranks, backend=backend)
                local_rank = ranks.index(self.rank)
                local_world_size = len(ranks)
                ranks_in_group = ranks

        return {
            "local_rank": local_rank,
            "process_group": process_group,
            "local_world_size": local_world_size,
            "ranks_in_group": ranks_in_group,
            "parallel_mode": parallel_mode,
        }
