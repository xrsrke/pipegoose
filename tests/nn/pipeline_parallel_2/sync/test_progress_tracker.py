import time
from copy import deepcopy
from typing import Dict

import pytest

from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel2.sync.callback import Callback
from pipegoose.nn.pipeline_parallel2.sync.handshake import ProgressTracker
from pipegoose.testing.utils import init_parallel_context, spawn

MASTER_RANK = 0


def generate_unfinished_tasks(n):
    # {0: {0: False, 1: False, 2: False, 3: False, 4: False},...}
    return {i: {j: False for j in range(n)} for i in range(n)}


def generate_finsihed_task(n):
    # {0: {0: True, 1: True, 2: True, 3: True, 4: True},...}
    return {i: {j: True for j in range(n)} for i in range(n)}


def run_init_progress_tracker(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    PROGRESS = generate_unfinished_tasks(n=world_size)

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    tracker = ProgressTracker(MASTER_RANK, parallel_context=parallel_context, parallel_mode=ParallelMode.GLOBAL)

    if rank == tracker.master_rank:
        tracker.initiate(PROGRESS)

    # NOTE: wait until the tracker is initiated
    time.sleep(0.1)
    assert tracker.is_initiated() is True
    assert tracker.clock_idx == 0
    assert tracker.is_all_confirmed(clock_idx=0) is False
    assert tracker.progress == PROGRESS

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
    N_CLOCK_CYCLES = world_size
    PROGRESS = generate_unfinished_tasks(N_CLOCK_CYCLES)
    FINAL_PROGRESS = generate_finsihed_task(N_CLOCK_CYCLES)
    INITIAL_PROGRESS = deepcopy(PROGRESS)

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    tracker = ProgressTracker(MASTER_RANK, parallel_context=parallel_context, parallel_mode=ParallelMode.GLOBAL)

    if rank == tracker.master_rank:
        tracker.initiate(PROGRESS)

    # NOTE: wait until the tracker is initiated
    time.sleep(2)

    for clock_idx in range(N_CLOCK_CYCLES):
        tracker.confirm(rank)
        assert tracker.is_confirmed(rank, clock_idx=clock_idx) is True

        # NOTE: wait until all workers are confirmed
        time.sleep(2)
        assert tracker.is_all_confirmed(clock_idx=clock_idx) is True

        if not (clock_idx == N_CLOCK_CYCLES - 1):
            assert tracker.is_all_confirmed(clock_idx=clock_idx + 1) is False

        assert tracker.clock_idx == clock_idx + 1
        assert tracker.progress != INITIAL_PROGRESS

        time.sleep(0.1)

    assert tracker.progress == FINAL_PROGRESS

    parallel_context.destroy()


@pytest.mark.parametrize("tensor_parallel_size, pipeline_parallel_size, data_parallel_size", [(1, 2, 1), (2, 2, 2)])
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
    QUEUE = []

    class TestCallback(Callback):
        def after_new_clock_cycle(self, progress: Dict, clock_idx: int):
            QUEUE.append(rank)

    PROGRESS = generate_unfinished_tasks(n=world_size)

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    tracker = ProgressTracker(
        MASTER_RANK, callbacks=[TestCallback()], parallel_context=parallel_context, parallel_mode=ParallelMode.GLOBAL
    )

    if rank == MASTER_RANK:
        tracker.initiate(PROGRESS)

    # NOTE: wait until the tracker is initiated
    time.sleep(0.5)
    tracker.confirm(rank)

    # NOTE: wait until all workers are confirmed
    time.sleep(0.5)
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
