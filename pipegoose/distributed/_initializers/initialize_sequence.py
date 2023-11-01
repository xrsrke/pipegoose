import torch.distributed as dist

from pipegoose.distributed._initializers.initializer import (
    ProcessGroupInitializer,
    ProcessGroupResult,
)
from pipegoose.distributed.parallel_mode import ParallelMode


class SequenceParallelGroupInitializer(ProcessGroupInitializer):
    def init_dist_group(self) -> ProcessGroupResult:
        num_sequence_parallel_groups = self.world_size // self.sequence_parallel_size

        local_rank = None
        process_group = None
        local_world_size = None
        ranks_in_group = None
        parallel_mode = ParallelMode.SEQUENCE

        for i in range(num_sequence_parallel_groups):
            ranks = range(i * self.sequence_parallel_size, (i + 1) * self.sequence_parallel_size)
            group = dist.new_group(ranks=ranks)

            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                local_world_size = len(ranks)
                ranks_in_group = ranks
                process_group = group

        return {
            "local_rank": local_rank,
            "process_group": process_group,
            "local_world_size": local_world_size,
            "ranks_in_group": ranks_in_group,
            "parallel_mode": parallel_mode,
        }
