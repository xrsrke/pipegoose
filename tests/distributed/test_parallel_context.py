import pytest
import torch

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.testing.utils import spawn

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

backend = ["gloo", pytest.param("nccl", marks=skip_if_no_cuda)]


def init_parallel_context(
    rank, world_size, seed, backend, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
):
    parallel_context = ParallelContext(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        host="localhost",
        port=port,
        seed=seed,
        backend=backend,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )

    parallel_modes = [
        ParallelMode.GLOBAL,
        ParallelMode.TENSOR,
        ParallelMode.PIPELINE,
        ParallelMode.DATA,
    ]

    assert parallel_context.tensor_parallel_size == tensor_parallel_size
    assert parallel_context.pipeline_parallel_size == pipeline_parallel_size
    assert parallel_context.data_parallel_size == data_parallel_size

    assert parallel_context.get_global_rank() == rank

    for parallel_mode in parallel_modes:
        # assert parallel_context.is_initialized(parallel_mode) is True
        assert type(parallel_context.get_local_rank(parallel_mode)) == int
        assert type(parallel_context.get_world_size(parallel_mode)) == int
        # assert isinstance(parallel_context.get_group(parallel_mode), ProcessGroup)
        assert isinstance(parallel_context.get_ranks_in_group(parallel_mode), list)

    # parallel_context.destroy()
    torch.distributed.destroy_process_group()

    # for parallel_mode in parallel_modes:
    #     assert parallel_context.is_initialized(parallel_mode) is False


@pytest.mark.parametrize(
    "world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size", [(1, 1, 1, 1), (8, 2, 2, 2)]
)
def test_init_parallel_context(world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    SEED = 69
    BACKEND = "gloo"

    spawn(
        init_parallel_context,
        nprocs=world_size,
        seed=SEED,
        backend=BACKEND,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )
