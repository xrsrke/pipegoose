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


# @pytest.fixture(scope="module")
def init_parallel_context(rank, world_size, port):
    parallel_context = ParallelContext(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        host="localhost",
        port=port,
        seed=69,
        backend="gloo",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
    )

    return parallel_context


def run_scatter(rank, world_size, port):
    parallel_context = init_parallel_context(rank, world_size, port)
    world_size = parallel_context.get_world_size(ParallelMode.GLOBAL)
    rank = parallel_context.get_local_rank(ParallelMode.GLOBAL)

    DIM = -1
    xs = torch.randn(2, world_size, dtype=torch.float32)
    temp = xs.clone()

    x = scatter(
        xs,
        dim=DIM,
        parallel_context=parallel_context,
        parallel_mode=ParallelMode.GLOBAL,
    )

    assert isinstance(x, torch.Tensor)
    # assert x.size() == temp.siz
    assert torch.equal(x, torch.chunk(temp, world_size, dim=DIM)[rank])
    assert x.dtype == temp.dtype
    assert x.requires_grad == temp.requires_grad


def test_scatter():
    spawn(run_scatter, nprocs=1)


def test_reduce(parallel_context):
    world_size = parallel_context.get_world_size(ParallelMode.GLOBAL)
    rank = parallel_context.get_local_rank(ParallelMode.GLOBAL)
    x = torch.randn(1, dtype=torch.float32)
    temp = x.clone()

    x = reduce(
        tensor=x,
        dst=0,
        parallel_context=parallel_context,
        parallel_mode=ParallelMode.GLOBAL,
    )

    if rank == 0:
        assert x == temp * world_size
        assert x.dtype == temp.dtype
        assert x.requires_grad == temp.requires_grad


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


def test_all_gather(parallel_context):
    rank = parallel_context.get_local_rank(ParallelMode.GLOBAL)

    tensor = torch.tensor([rank * 1.0])

    gathered_tensors = all_gather(tensor, dim=0, parallel_context=parallel_context, parallel_mode=ParallelMode.GLOBAL)

    for i, gathered_tensor in enumerate(gathered_tensors):
        assert torch.equal(gathered_tensor, torch.tensor([i * 1.0]))


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


def test_reduce_scatter(parallel_context):
    pass


# if __name__ == "__main__":
#     spawn(run_scatter, nprocs=1)
