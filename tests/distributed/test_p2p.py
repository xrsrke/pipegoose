import torch

from pipegoose.distributed.functional import recv, send
from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.testing.utils import spawn


def init_parallel_context(rank, world_size, port, pipeline_parallel_size):
    TENSOR_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    parallel_context = ParallelContext(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        host="localhost",
        port=port,
        seed=69,
        backend="gloo",
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=DATA_PARALLEL_SIZE,
    )

    return parallel_context


def run_p2p(rank, world_size, port, pipeline_parallel_size):
    parallel_context = init_parallel_context(rank, world_size, port, pipeline_parallel_size)
    rank = parallel_context.get_local_rank(parallel_mode=ParallelMode.PIPELINE)

    data = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, dtype=torch.float)
    send(data, src=0, dst=1, parallel_context=parallel_context)

    received_data = recv(src=0, dst=1, parallel_context=parallel_context)

    if rank == 1:
        assert torch.allclose(data, received_data)
        assert received_data.requires_grad == data.requires_grad
        assert received_data.dtype == data.dtype
    else:
        assert received_data is None


def test_send_recv_p2p():
    spawn(
        run_p2p,
        world_size=2,
        pipeline_parallel_size=2,
    )
