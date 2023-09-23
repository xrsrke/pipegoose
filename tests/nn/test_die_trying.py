import time

import torch
import torch.distributed.rpc as rpc

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel2.sync.func import get_execution_plan
from pipegoose.nn.pipeline_parallel2.sync.handshake import SchedulerHandshake
from pipegoose.testing.utils import spawn

RPC_RECEIVE_QUEUE = list()


def recv_rpc_call(value):
    tensor = torch.Tensor(value)
    RPC_RECEIVE_QUEUE.append(tensor)


def run_send_rcv_rpc(rank, world_size, seed, backend, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    MICROBATCH_IDX = 0
    EXPECTED_TASKS = {}

    for partition_idx in range(pipeline_parallel_size):
        EXPECTED_TASKS[(MICROBATCH_IDX, partition_idx)] = False

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

    if rank == 0:
        # task = torch.tensor([1, 2])
        handshake = SchedulerHandshake(parallel_context, ParallelMode.GLOBAL)
        handshake.initiate(EXPECTED_TASKS)
        # rpc.rpc_sync(to=parallel_context.get_worker_name(rank=1), func=recv_execution_plan, args=(task,))

    else:
        time.sleep(2)
        get_execution_plan()[(1, 2)]
        # assert tensor is False

    parallel_context.destroy()

    if pipeline_parallel_size > 1:
        assert rpc._is_current_rpc_agent_set() is False


def test_send_rcv_rpc():
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
    )
