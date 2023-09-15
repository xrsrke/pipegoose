import pytest
import torch

from pipegoose.distributed.functional import (
    all_gather,
    all_reduce,
    broadcast,
    reduce,
    scatter,
)
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.testing.utils import init_parallel_context, spawn


@pytest.fixture
def parallel_modes():
    return [ParallelMode.GLOBAL, ParallelMode.TENSOR, ParallelMode.PIPELINE, ParallelMode.DATA]


PARAMETRIZE = pytest.mark.parametrize(
    "world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size", [(1, 1, 1, 1), (8, 2, 2, 2)]
)


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


def scatter_logic(rank, ranks_in_group, parallel_context, parallel_mode):
    if rank in ranks_in_group:
        DIM = -1
        world_size = parallel_context.get_world_size(parallel_mode)
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


@PARAMETRIZE
def test_scatter(world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, parallel_modes):
    spawn(
        run_parallel_test,
        test_logic=scatter_logic,
        world_size=world_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        parallel_modes=parallel_modes,
    )


def reduce_logic(rank, ranks_in_group, parallel_context, parallel_mode):
    if rank in ranks_in_group:
        world_size = parallel_context.get_world_size(parallel_mode)
        dst = parallel_context.get_ranks_in_group(parallel_mode)[-1]

        x = torch.tensor(1.0, dtype=torch.float32)
        expected_output = x.clone() * world_size

        reduce(
            tensor=x,
            dst=dst,
            parallel_context=parallel_context,
            parallel_mode=parallel_mode,
        )

        if rank == dst:
            assert torch.equal(x, expected_output)
            assert x.dtype == expected_output.dtype
            assert x.requires_grad == expected_output.requires_grad


@PARAMETRIZE
def test_reduce(world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, parallel_modes):
    spawn(
        run_parallel_test,
        test_logic=reduce_logic,
        world_size=world_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        parallel_modes=parallel_modes,
    )


def run_broadcast(rank, world_size, port, parallel_modes, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    for parallel_mode in parallel_modes:
        rank = parallel_context.get_local_rank(parallel_mode)
        ranks_in_group = parallel_context.get_ranks_in_group(parallel_mode)

        if rank == ranks_in_group:
            src = parallel_context.get_ranks_in_group(parallel_mode)[-1]
            if rank == src:
                x = torch.tensor(6.9, dtype=torch.float32, requires_grad=True)
            else:
                x = torch.tensor(4.2, dtype=torch.float32)

            broadcast(x, src=src, parallel_context=parallel_context, parallel_mode=parallel_mode)

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


def run_all_gather(rank, world_size, port, parallel_modes, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    for parallel_mode in parallel_modes:
        rank = parallel_context.get_local_rank(parallel_mode)
        ranks_in_group = parallel_context.get_ranks_in_group(parallel_mode)

        if rank in ranks_in_group:
            world_size = parallel_context.get_world_size(parallel_mode)
            tensor = torch.tensor(rank, dtype=torch.float32, requires_grad=True)
            expected_output = torch.tensor(ranks_in_group, dtype=torch.float32, requires_grad=True)

            output = all_gather(tensor, dim=0, parallel_context=parallel_context, parallel_mode=parallel_mode)

            assert torch.allclose(output, expected_output)
            assert output.dtype == expected_output.dtype
            # TODO: do we need to check this?
            # assert output.requires_grad == expected_output.requires_grad

    parallel_context.destroy()


@pytest.mark.parametrize(
    "world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size", [(1, 1, 1, 1), (8, 2, 2, 2)]
)
def test_all_gather(parallel_modes, world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    spawn(
        run_all_gather,
        world_size=world_size,
        parallel_modes=parallel_modes,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )


def run_all_reduce(rank, world_size, port, parallel_modes, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    for parallel_mode in parallel_modes:
        rank = parallel_context.get_local_rank(parallel_mode)
        ranks_in_group = parallel_context.get_ranks_in_group(parallel_mode)

        if rank in ranks_in_group:
            x = torch.tensor(rank, dtype=torch.float32)
            temp = x.clone()

            all_reduce(
                tensor=x,
                parallel_context=parallel_context,
                parallel_mode=parallel_mode,
            )

            assert x == sum(ranks_in_group)
            assert x.dtype == temp.dtype
            assert x.requires_grad == temp.requires_grad

    parallel_context.destroy()


@pytest.mark.skip(reason="not implemented")
def test_reduce_scatter(parallel_context):
    pass


def all_reduce_logic(rank, ranks_in_group, parallel_context, parallel_mode):
    if rank in ranks_in_group:
        x = torch.tensor(rank, dtype=torch.float32)
        temp = x.clone()

        all_reduce(
            tensor=x,
            parallel_context=parallel_context,
            parallel_mode=parallel_mode,
        )

        assert x == sum(ranks_in_group)
        assert x.dtype == temp.dtype
        assert x.requires_grad == temp.requires_grad


@PARAMETRIZE
def test_all_reduce(world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, parallel_modes):
    spawn(
        run_parallel_test,
        test_logic=all_reduce_logic,
        world_size=world_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        parallel_modes=parallel_modes,
    )
