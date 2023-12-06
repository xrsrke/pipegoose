import time

import pytest
import torch
import torch.distributed.rpc as rpc

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.testing.utils import spawn

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

backend = ["gloo", pytest.param("nccl", marks=skip_if_no_cuda)]


RPC_RECEIVE_QUEUE = list()


def recv_rpc_call(value):
    tensor = torch.Tensor(value)
    RPC_RECEIVE_QUEUE.append(tensor)


def run_send_rcv_rpc(
    rank, world_size, seed, backend, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, rpc_type
):
    VALUE = 69

    RPC_TYPE_TO_FUNC = {"rpc_sync": rpc.rpc_sync, "rpc_async": rpc.rpc_async}
    rpc_func = RPC_TYPE_TO_FUNC[rpc_type]

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

    assert isinstance(parallel_context.get_worker_name(rank), str)

    if world_size > 1:
        assert rpc._is_current_rpc_agent_set() is True

    if rank == 0:
        tensor = torch.tensor(VALUE)

        fut = rpc_func(to=parallel_context.get_worker_name(rank=1), func=recv_rpc_call, args=(tensor,))

        if rpc_func == rpc.rpc_async:
            fut.wait()

    else:
        while len(RPC_RECEIVE_QUEUE) < 1:
            time.sleep(0.1)

        tensor = RPC_RECEIVE_QUEUE.pop()

        assert tensor == VALUE

    parallel_context.destroy()

    if world_size > 1:
        assert rpc._is_current_rpc_agent_set() is False


@pytest.mark.parametrize("rpc_type", ["rpc_sync", "rpc_async"])
def test_send_rcv_rpc(rpc_type):
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 2
    DATA_PARALLEL_SIZE = 1

    SEED = 69
    BACKEND = "gloo"

    spawn(
        run_send_rcv_rpc,
        world_size=PIPELINE_PARALLEL_SIZE,
        seed=SEED,
        backend=BACKEND,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        rpc_type=rpc_type,
    )
