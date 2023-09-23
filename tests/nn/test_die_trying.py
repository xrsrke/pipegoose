import time

import pytest

from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel2.sync.handshake import SchedulerHandshake
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
        handshake.initiate(EXPECTED_TASKS)
        assert handshake.is_initiated() is True

    else:
        time.sleep(2)
        assert handshake.is_initiated() is True

        output = handshake.pipeline_progress
        assert output == EXPECTED_TASKS

    parallel_context.destroy()


@pytest.mark.parametrize("tensor_parallel_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [2])
def test_send_rcv_rpc(tensor_parallel_size, pipeline_parallel_size):
    DATA_PARALLEL_SIZE = 1

    world_size = tensor_parallel_size * pipeline_parallel_size * DATA_PARALLEL_SIZE

    spawn(
        run_send_rcv_rpc,
        world_size=world_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=DATA_PARALLEL_SIZE,
    )
