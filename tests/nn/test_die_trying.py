import time
from copy import deepcopy

import pytest

from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel2._utils import get_partition_idx
from pipegoose.nn.pipeline_parallel2.sync.handshake import SchedulerHandshake
from pipegoose.testing.utils import init_parallel_context, spawn


def get_gpipe_schedules(n_partitions, n_microbatches):
    n_clock_cycles = n_partitions + n_microbatches - 1
    schedules = []
    for clock_idx in range(n_clock_cycles):
        start_partrition = max(clock_idx + 1 - n_microbatches, 0)
        end_partition = min(clock_idx + 1, n_partitions)

        tasks = []
        for partition_idx in range(start_partrition, end_partition):
            microbatch_idx = clock_idx - partition_idx
            tasks.append((microbatch_idx, partition_idx))

        schedules.append(tasks)

    return schedules


def schedules_to_progress(schedules):
    return {i: {item: False for item in sublist} for i, sublist in enumerate(schedules)}


def run_send_rcv_rpc(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    N_MICROBATCHES = 4
    MICROBATCH_IDX = 0

    schedules = get_gpipe_schedules(pipeline_parallel_size, N_MICROBATCHES)
    PROGRESS = schedules_to_progress(schedules)

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    handshake = SchedulerHandshake(parallel_context, ParallelMode.GLOBAL)

    if rank == SchedulerHandshake.MASTER_RANK:
        handshake.initiate(PROGRESS)
        assert handshake.is_initiated() is True
        assert handshake.progress == PROGRESS
        assert handshake.clock_idx == 0

        # NOTE: wait until all workers are confirmed
        time.sleep(5)
        assert SchedulerHandshake.is_all_confirmed(clock_idx=0) is True

        # NOTE: after all workers are confirmed,
        # the clock index should be incremented
        assert handshake.clock_idx == 1
    else:
        # NOTE: wait until the handshake is initiated
        time.sleep(2)
        assert handshake.is_initiated() is True
        assert handshake.progress == PROGRESS
        assert handshake.clock_idx == 0

        PREV_CLOCK_IDX = deepcopy(handshake.clock_idx)
        task = (MICROBATCH_IDX, get_partition_idx(parallel_context))
        handshake.confirm(task)
        assert handshake.is_confirmed(task, PREV_CLOCK_IDX) is True

        # NOTE: wait until all workers are confirmed
        # time.sleep(5)
        # assert handshake.clock_idx == PREV_CLOCK_IDX + 1

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
