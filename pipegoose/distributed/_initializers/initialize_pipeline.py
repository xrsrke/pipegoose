from pipegoose.distributed._initializers.initializer import (
    ProcessGroupInitializer,
    ProcessGroupResult,
)
from pipegoose.distributed.parallel_mode import ParallelMode


class PipelineParallelGroupInitializer(ProcessGroupInitializer):
    def init_dist_group(self) -> ProcessGroupResult:
        # backend = dist.get_backend()
        num_pipeline_parallel_groups = self.world_size // self.pipeline_parallel_size
        local_rank = None
        local_world_size = None
        ranks_in_group = None
        parallel_mode = ParallelMode.PIPELINE

        for i in range(num_pipeline_parallel_groups):
            ranks = list(range(i, self.world_size, num_pipeline_parallel_groups))

            if self.rank in ranks:
                # process_group = dist.new_group(ranks=ranks, backend=backend)
                local_rank = ranks.index(self.rank)
                local_world_size = len(ranks)
                ranks_in_group = ranks

        return {
            "local_rank": local_rank,
            # "process_group": process_group,
            "local_world_size": local_world_size,
            "ranks_in_group": ranks_in_group,
            "parallel_mode": parallel_mode,
        }
