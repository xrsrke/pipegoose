import torch.distributed as dist

from pipegoose.distributed._initializers.initializer import (
    ProcessGroupInitializer,
    ProcessGroupResult,
)
from pipegoose.distributed.context import ParallelMode


class DataParallelGroupInitializer(ProcessGroupInitializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_pipeline_parallel_groups = self.world_size // self.pipeline_parallel_size

    def init_dist_group(self) -> ProcessGroupResult:
        local_rank = None
        process_group = None
        group_world_size = None
        ranks_in_groups = None
        mode = ParallelMode.DATA

        for i in range(self.pipeline_parallel_size):
            start_rank = i * self.num_pipeline_parallel_groups
            end_rank = (i + 1) * self.num_pipeline_parallel_groups

            for j in range(self.tensor_parallel_size):
                ranks = range(start_rank + j, end_rank, self.tensor_parallel_size)

                if self.rank in ranks:
                    process_group = dist.new_group(ranks=ranks)
                    local_rank = ranks.index(self.rank)
                    group_world_size = len(ranks)
                    ranks_in_groups = ranks

        return {
            "local_rank": local_rank,
            "process_group": process_group,
            "group_world_size": group_world_size,
            "ranks_in_group": ranks_in_groups,
            "mode": mode,
        }
