import os

import torch
from torch.distributed import ProcessGroup

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode


def run_worker(seed, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    parallel_context = ParallelContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=local_world_size,
        host=host,
        port=port,
        backend="gloo",
        seed=seed,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )

    assert parallel_context.tensor_parallel_size == TENSOR_PARALLEL_SIZE
    assert parallel_context.pipeline_parallel_size == PIPELINE_PARALLEL_SIZE
    assert parallel_context.data_parallel_size == DATA_PARALLEL_SIZE

    assert parallel_context.get_global_rank() == rank

    parallel_modes = [
        ParallelMode.GLOBAL,
        ParallelMode.TENSOR,
        # ParallelMode.PIPELINE,
        # ParallelMode.DATA,
    ]

    for parallel_mode in parallel_modes:
        assert parallel_context.is_initialized(parallel_mode) is True
        assert type(parallel_context.get_local_rank(parallel_mode)) == int
        assert type(parallel_context.get_world_size(parallel_mode)) == int
        assert isinstance(parallel_context.get_group(parallel_mode), ProcessGroup)
        assert isinstance(parallel_context.get_ranks_in_group(parallel_mode), list)

    if torch.cuda.is_available():
        assert isinstance(torch.cuda.current_device(), int)

    # # TODO: test seed

    parallel_context.destroy()

    for parallel_mode in parallel_modes:
        assert parallel_context.is_initialized(parallel_mode) is False


if __name__ == "__main__":
    TENSOR_PARALLEL_SIZE = 2
    PIPELINE_PARALLEL_SIZE = 2
    DATA_PARALLEL_SIZE = 2
    SEED = 69
    WORLD_SIZE = 8

    run_worker(SEED, TENSOR_PARALLEL_SIZE, PIPELINE_PARALLEL_SIZE, DATA_PARALLEL_SIZE)
