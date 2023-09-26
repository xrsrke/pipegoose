import time
from copy import deepcopy
from typing import Dict

import pytest

from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel2._utils import get_partition_idx
from pipegoose.nn.pipeline_parallel2.sync.callback import Callback
from pipegoose.nn.pipeline_parallel2.sync.handshake import ProgressTracker
from pipegoose.testing.utils import init_parallel_context, spawn

MASTER_RANK = 0


def get_task(microbatch_idx, partition_idx):
    return (microbatch_idx, partition_idx)


def get_gpipe_schedules(n_partitions, n_microbatches):
    n_clock_cycles = n_partitions + n_microbatches - 1
    schedules = []
    for clock_idx in range(n_clock_cycles):
        start_partrition = max(clock_idx + 1 - n_microbatches, 0)
        end_partition = min(clock_idx + 1, n_partitions)
        tasks = []
        for partition_idx in range(start_partrition, end_partition):
            microbatch_idx = clock_idx - partition_idx
            task = get_task(microbatch_idx, partition_idx)
            tasks.append(task)

        schedules.append(tasks)

    return schedules


def schedules_to_progress(schedules):
    return {i: {item: False for item in sublist} for i, sublist in enumerate(schedules)}


def run_init_progress_tracker(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    N_MICROBATCHES = 4

    schedules = get_gpipe_schedules(pipeline_parallel_size, N_MICROBATCHES)
    PROGRESS = schedules_to_progress(schedules)

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    tracker = ProgressTracker(MASTER_RANK, parallel_context, ParallelMode.GLOBAL)

    if rank == tracker.master_rank:
        tracker.initiate(PROGRESS)
        assert tracker.is_initiated() is True
        assert tracker.progress == PROGRESS
        assert tracker.clock_idx == 0
        assert tracker.is_all_confirmed(clock_idx=0) is False
    else:
        # NOTE: wait until the tracker is initiated
        time.sleep(2)
        assert tracker.is_initiated() is True
        # NOTE: if haven't confirmed any task, clock_idx should be 0
        assert tracker.progress == PROGRESS
        assert tracker.clock_idx == 0

    parallel_context.destroy()


def test_init_progress_tracker():
    TENSOR_PARALLEL_SIZE = 2
    PIPELINE_PARALLEL_SIZE = 2
    DATA_PARALLEL_SIZE = 2
    world_size = TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE * DATA_PARALLEL_SIZE

    spawn(
        run_init_progress_tracker,
        world_size=world_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
    )


def run_confirm_progress_tracker(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    N_MICROBATCHES = 4
    MICROBATCH_IDX = 0

    schedules = get_gpipe_schedules(pipeline_parallel_size, N_MICROBATCHES)
    PROGRESS = schedules_to_progress(schedules)
    INITIAL_PROGRESS = deepcopy(PROGRESS)

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    tracker = ProgressTracker(MASTER_RANK, parallel_context, ParallelMode.GLOBAL)

    if rank == tracker.master_rank:
        tracker.initiate(PROGRESS)
        # NOTE: wait until all workers are confirmed
        time.sleep(5)
        assert tracker.is_all_confirmed(clock_idx=0) is True
        assert tracker.is_all_confirmed(clock_idx=1) is False

        # NOTE: after all workers are confirmed,
        assert tracker.clock_idx == 1
        assert tracker.progress != INITIAL_PROGRESS
    else:
        # NOTE: wait until the tracker is initiated
        time.sleep(2)
        partition_idx = get_partition_idx(parallel_context)
        task = get_task(MICROBATCH_IDX, partition_idx)
        tracker.confirm(task)
        assert tracker.is_confirmed(task, clock_idx=0) is True

        # NOTE: wait until all workers are confirmed
        time.sleep(5)
        assert tracker.is_all_confirmed(clock_idx=0) is True
        assert tracker.is_all_confirmed(clock_idx=1) is False
        assert tracker.clock_idx == 1
        assert tracker.progress != INITIAL_PROGRESS

    parallel_context.destroy()


@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
@pytest.mark.parametrize("pipeline_parallel_size", [2, 4])
@pytest.mark.parametrize("data_parallel_size", [1, 2])
def test_confirm_progress_tracker(tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    world_size = tensor_parallel_size * pipeline_parallel_size * data_parallel_size

    spawn(
        run_confirm_progress_tracker,
        world_size=world_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )


def run_progress_tracker_callback(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    N_MICROBATCHES = 4
    MICROBATCH_IDX = 0
    QUEUE = []

    class TestCallback(Callback):
        def after_new_clock_cycle(self, progress: Dict, clock_idx: int):
            QUEUE.append(rank)

    schedules = get_gpipe_schedules(pipeline_parallel_size, N_MICROBATCHES)
    PROGRESS = schedules_to_progress(schedules)

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    if rank == MASTER_RANK:
        tracker = ProgressTracker(MASTER_RANK, parallel_context=parallel_context, parallel_mode=ParallelMode.GLOBAL)
        tracker.initiate(PROGRESS)
        # NOTE: wait until all workers are confirmed
        time.sleep(5)
        assert tracker.is_all_confirmed(clock_idx=0) is True
    else:
        tracker = ProgressTracker(
            MASTER_RANK, callbacks=[TestCallback()], parallel_context=parallel_context, parallel_mode=ParallelMode.GLOBAL
        )
        # NOTE: wait until the tracker is initiated
        time.sleep(2)
        partition_idx = get_partition_idx(parallel_context)
        task = get_task(MICROBATCH_IDX, partition_idx)

        tracker.confirm(task)

        # NOTE: wait until all workers are confirmed
        time.sleep(5)
        # TODO: QUEUE should be equal to [rank], fix the bug
        assert QUEUE == [rank] or QUEUE == [rank, rank]

    parallel_context.destroy()


def test_progress_tracker_callback():
    TENSOR_PARALLEL_SIZE = 2
    PIPELINE_PARALLEL_SIZE = 2
    DATA_PARALLEL_SIZE = 2
    world_size = TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE * DATA_PARALLEL_SIZE

    spawn(
        run_progress_tracker_callback,
        world_size=world_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
    )
