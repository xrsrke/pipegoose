import time

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode


def sleep(timeout: int = 0.05):
    time.sleep(timeout)


def get_partition_idx(parallel_context: ParallelContext) -> int:
    rank = parallel_context.get_global_rank()
    ranks_in_group = parallel_context.get_ranks_in_group(ParallelMode.PIPELINE)
    # pipeline_stage_idx = rank // n_ranks_per_group
    # return pipeline_stage_idx
    return ranks_in_group.index(rank)


def is_last_stage(parallel_context: ParallelContext) -> bool:
    partition_idx = get_partition_idx(parallel_context)
    n_stages = parallel_context.pipeline_parallel_size
    return partition_idx == (n_stages - 1)
