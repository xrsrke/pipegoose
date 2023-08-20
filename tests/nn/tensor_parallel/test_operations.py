import pytest
import torch

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.testing.utils import spawn


@pytest.fixture
def parallel_modes():
    return [ParallelMode.GLOBAL, ParallelMode.TENSOR, ParallelMode.PIPELINE, ParallelMode.DATA]


PARAMETRIZE = pytest.mark.parametrize(
    "world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size", [(1, 1, 1, 1), (8, 2, 2, 2)]
)


def init_parallel_context(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    parallel_context = ParallelContext(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        host="localhost",
        port=port,
        seed=69,
        backend="gloo",
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )

    return parallel_context


def run_parallel_test(
    rank, world_size, port, parallel_modes, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, test_logic
):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    for parallel_mode in parallel_modes:
        rank = parallel_context.get_local_rank(parallel_mode)
        ranks_in_group = parallel_context.get_ranks_in_group(parallel_mode)
        test_logic(rank, ranks_in_group, parallel_context, parallel_mode)

    parallel_context.destroy()


def run_broadcast(rank, world_size, port, parallel_modes, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    for parallel_mode in parallel_modes:
        rank = parallel_context.get_local_rank(parallel_mode)
        ranks_in_group = parallel_context.get_ranks_in_group(parallel_mode)

        if rank == ranks_in_group:
            src_rank = parallel_context.get_ranks_in_group(parallel_mode)[-1]
            if rank == src_rank:
                x = torch.tensor(6.9, dtype=torch.float32, requires_grad=True)
            else:
                x = torch.tensor(4.2, dtype=torch.float32)

            # Broadcast.apply(x, src=src_rank, parallel_context=parallel_context, parallel_mode=parallel_mode)

            assert torch.equal(x, torch.tensor(6.9))
            assert x.dtype == torch.float32
            assert x.requires_grad is True

    parallel_context.destroy()


@pytest.mark.parametrize(
    "world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size", [(1, 1, 1, 1), (8, 2, 2, 2)]
)
def test_broadcast(parallel_modes, world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    spawn(
        run_broadcast,
        world_size=world_size,
        parallel_modes=parallel_modes,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )
