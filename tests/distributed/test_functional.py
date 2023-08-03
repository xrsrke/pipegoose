import pytest
import torch

from pipegoose.distributed.functional import (
    all_gather,
    all_reduce,
    broadcast,
    reduce,
    scatter,
)
from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.testing.utils import spawn


@pytest.fixture
def parallel_modes():
    return [ParallelMode.GLOBAL, ParallelMode.TENSOR, ParallelMode.PIPELINE, ParallelMode.DATA]


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


def run_scatter(rank, world_size, port, parallel_modes, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    for parallel_mode in parallel_modes:
        world_size = parallel_context.get_world_size(parallel_mode)
        rank = parallel_context.get_local_rank(parallel_mode)

        DIM = -1
        xs = torch.randn(2, world_size, dtype=torch.float32)
        expected = torch.chunk(xs.clone(), world_size, dim=DIM)[rank]

        x = scatter(
            xs,
            dim=DIM,
            parallel_context=parallel_context,
            parallel_mode=parallel_mode,
        )

        assert isinstance(x, torch.Tensor)
        assert x.size() == expected.shape
        assert torch.equal(x, expected)
        assert x.dtype == expected.dtype
        assert x.requires_grad == expected.requires_grad

    parallel_context.destroy()


@pytest.mark.parametrize(
    "world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size", [(1, 1, 1, 1), (8, 2, 2, 2)]
)
def test_scatter(parallel_modes, world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    spawn(
        run_scatter,
        nprocs=world_size,
        parallel_modes=parallel_modes,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )


def run_reduce(rank, world_size, port, parallel_modes, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    for parallel_mode in parallel_modes:
        world_size = parallel_context.get_world_size(parallel_mode)
        rank = parallel_context.get_local_rank(parallel_mode)
        dst_rank = parallel_context.get_ranks_in_group(parallel_mode)[-1]

        x = torch.tensor(1.0, dtype=torch.float32)
        expected_output = x.clone() * world_size

        torch.distributed.barrier()
        output = reduce(
            tensor=x,
            dst=dst_rank,
            parallel_context=parallel_context,
            parallel_mode=parallel_mode,
        )

        if rank == dst_rank:
            assert torch.equal(output, expected_output)
            assert output.dtype == expected_output.dtype
            assert output.requires_grad == expected_output.requires_grad

    parallel_context.destroy()


@pytest.mark.parametrize(
    "world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size", [(1, 1, 1, 1), (8, 2, 2, 2)]
)
def test_reduce(parallel_modes, world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    spawn(
        run_reduce,
        nprocs=world_size,
        parallel_modes=parallel_modes,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )


@pytest.mark.skip(reason="not implemented")
def test_broadcast(parallel_context):
    rank = parallel_context.get_local_rank(ParallelMode.GLOBAL)
    if rank == 0:
        x = torch.tensor(6.9, dtype=torch.float16, requires_grad=True)
    else:
        x = torch.tensor(4.2, dtype=torch.float32)

    broadcast(x, src=0, parallel_context=parallel_context, parallel_mode=ParallelMode.GLOBAL)

    assert torch.equal(x, torch.tensor(6.9))
    assert x.dtype == torch.float16
    assert x.requires_grad is True


@pytest.mark.skip(reason="not implemented")
def test_all_gather(parallel_context):
    rank = parallel_context.get_local_rank(ParallelMode.GLOBAL)

    tensor = torch.tensor([rank * 1.0])

    gathered_tensors = all_gather(tensor, dim=0, parallel_context=parallel_context, parallel_mode=ParallelMode.GLOBAL)

    for i, gathered_tensor in enumerate(gathered_tensors):
        assert torch.equal(gathered_tensor, torch.tensor([i * 1.0]))


@pytest.mark.skip(reason="not implemented")
def test_all_reduce(parallel_context):
    world_size = parallel_context.get_world_size(ParallelMode.GLOBAL)

    x = torch.randn(1, dtype=torch.float32)
    temp = x.clone()

    output = all_reduce(
        tensor=x,
        parallel_context=parallel_context,
        parallel_mode=ParallelMode.GLOBAL,
    )

    assert output == temp * world_size
    assert output.dtype == temp.dtype
    assert output.requires_grad == temp.requires_grad


@pytest.mark.skip(reason="not implemented")
def test_reduce_scatter(parallel_context):
    pass


# if __name__ == "__main__":
#     spawn(run_scatter, nprocs=1)
