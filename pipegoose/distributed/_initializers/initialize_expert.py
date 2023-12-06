import torch.distributed as dist

from pipegoose.distributed._initializers.initializer import (
    ProcessGroupInitializer,
    ProcessGroupResult,
)
from pipegoose.distributed.parallel_mode import ParallelMode


class ExpertDataParallelGroupInitializer(ProcessGroupInitializer):
    """
    Initialize the process group for data parallelism in expert parallelism.

    Pipeline MoE: A Flexible MoE Implementation with Pipeline Parallelism" by Xin Chen et al
    https://arxiv.org/abs/2304.11414

    NOTE: This looks similar to TensorParallelGroupInitializer, because it aligns with the paper.
    """

    def init_dist_group(self) -> ProcessGroupResult:
        num_tensor_parallel_groups = self.world_size // self.tensor_parallel_size
        local_rank = None
        process_group = None
        local_world_size = None
        ranks_in_group = None
        parallel_mode = ParallelMode.EXPERT_DATA

        for i in range(num_tensor_parallel_groups):
            ranks = list(range(i * self.tensor_parallel_size, (i + 1) * self.tensor_parallel_size))
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
