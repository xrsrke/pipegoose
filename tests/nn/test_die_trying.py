import time

from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel2.sync.handshake import (
    SchedulerHandshake,
    get_execution_plan,
)
from pipegoose.testing.utils import init_parallel_context, spawn


def run_send_rcv_rpc(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    MICROBATCH_IDX = 0
    EXPECTED_TASKS = {}

    for partition_idx in range(pipeline_parallel_size):
        EXPECTED_TASKS[(MICROBATCH_IDX, partition_idx)] = False

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    handshake = SchedulerHandshake(parallel_context, ParallelMode.GLOBAL)

    if rank == 0:
        # task = torch.tensor([1, 2])
        handshake.initiate(EXPECTED_TASKS)
        assert handshake.is_initiated() is True
        # rpc.rpc_sync(to=parallel_context.get_worker_name(rank=1), func=recv_execution_plan, args=(task,))

    else:
        time.sleep(2)
        # # handshake = SchedulerHandshake(parallel_context, ParallelMode.GLOBAL)
        # # assert handshake.is_initiated() is True
        output = get_execution_plan()
        assert output == EXPECTED_TASKS

    parallel_context.destroy()


def test_send_rcv_rpc():
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 2
    DATA_PARALLEL_SIZE = 1

    spawn(
        run_send_rcv_rpc,
        world_size=PIPELINE_PARALLEL_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
    )
