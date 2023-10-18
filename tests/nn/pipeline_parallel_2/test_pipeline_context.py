import threading

import pytest

from pipegoose.nn.pipeline_parallel2._comm import get_pipeline_context
from pipegoose.nn.pipeline_parallel2.pipeline_context import (
    PipelineContext,
    TrainingState,
)
from pipegoose.nn.pipeline_parallel2.scheduler import SchedulerType, get_scheduler
from pipegoose.nn.pipeline_parallel2.task import Task
from pipegoose.testing.utils import init_parallel_context, spawn

EXPECTED_SCHEDULES = [
    [(0, 0)],
    [(1, 0), (0, 1)],
    [(2, 0), (1, 1), (0, 2)],
    [(3, 0), (2, 1), (1, 2), (0, 3)],
    [(3, 1), (2, 2), (1, 3), (0, 4)],
    [(3, 2), (2, 3), (1, 4)],
    [(3, 3), (2, 4)],
    [(3, 4)],
]


def test_get_pipeline_schedule_from_training_state(parallel_context):
    N_PARTITIONS = 5
    N_MICROBATCHES = 4

    scheduler = get_scheduler(SchedulerType.GPIPE)(N_MICROBATCHES, N_PARTITIONS)

    pipeline_context = PipelineContext(scheduler, parallel_context)

    assert pipeline_context.state == TrainingState.IDLE

    pipeline_context.forward()
    assert pipeline_context.state == TrainingState.FORWARD
    assert len(pipeline_context.schedule) > 0

    pipeline_context.backward()
    assert pipeline_context.state == TrainingState.BACKWARD
    # assert len(pipeline_context.schedule) > 0

    pipeline_context.finish()
    assert pipeline_context.state == TrainingState.FINISHED


def run_pipeline_context(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    N_PARTITIONS = 5
    N_MICROBATCHES = 4

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    scheduler = get_scheduler(SchedulerType.GPIPE)(N_MICROBATCHES, N_PARTITIONS)

    pipeline_context = PipelineContext(scheduler, parallel_context)
    pipeline_context.forward()

    assert get_pipeline_context() == pipeline_context
    assert isinstance(pipeline_context.partition_idx, int)

    assert pipeline_context.clock_idx == 0
    assert isinstance(pipeline_context.schedule, list)
    assert isinstance(pipeline_context.schedules, list)

    assert pipeline_context.num_microbatches == N_MICROBATCHES

    # assert isinstance(pipeline_context.get_schedule_from_partition(clock_idx=3, partition_idx=2), list)
    # assert isinstance(pipeline_context.get_schedule_from_microbatch(clock_idx=3, microbatch_idx=0), list)

    CURRENT_CLOCK_IDX = pipeline_context.clock_idx
    pipeline_context.increase_a_clock_cycle()

    assert pipeline_context.clock_idx == CURRENT_CLOCK_IDX + 1


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2, 4])
def test_run_pipeline_context(pipeline_parallel_size):
    TENSOR_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    spawn(
        run_pipeline_context,
        world_size=pipeline_parallel_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=DATA_PARALLEL_SIZE,
    )


def run_get_the_next_pipeline_schedule_from_pipeline_context(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
):
    N_PARTITIONS = 5
    N_MICROBATCHES = 4

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    scheduler = get_scheduler(SchedulerType.GPIPE)(N_MICROBATCHES, N_PARTITIONS)

    pipeline_context = PipelineContext(scheduler, parallel_context)
    pipeline_context.forward()

    next_schedules = pipeline_context.get_next_schedule_from_microbatch(microbatch_idx=0)
    assert isinstance(next_schedules, list)

    # NOTE: x is (microbatch_idx, partition_idx)
    # x[0] means microbatch_idx of the task
    EXPECTED_NEXT_SCHEDULE = [x for x in EXPECTED_SCHEDULES[pipeline_context.clock_idx + 1] if x[0] == 0]
    assert len(next_schedules) == len(EXPECTED_NEXT_SCHEDULE)
    for task, (_, partition_idx) in zip(next_schedules, EXPECTED_NEXT_SCHEDULE):
        assert task.partition_idx == partition_idx


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2, 4])
def test_get_the_next_pipeline_schedule_from_pipeline_context(pipeline_parallel_size):
    TENSOR_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    spawn(
        run_get_the_next_pipeline_schedule_from_pipeline_context,
        world_size=pipeline_parallel_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=DATA_PARALLEL_SIZE,
    )


def run_get_syncronous_schedule(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    N_PARTITIONS = 4
    N_MICROBATCHES = 5

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    scheduler = get_scheduler(SchedulerType.GPIPE)(N_MICROBATCHES, N_PARTITIONS)
    TOTAL_SCHEDULES = scheduler.total_clock_cycles

    def increase_clock_every_second(pipeline_context):
        for _ in range(TOTAL_SCHEDULES):
            pipeline_context.increase_a_clock_cycle()

    pipeline_context = PipelineContext(scheduler, parallel_context)
    pipeline_context.forward()
    clock_thread = threading.Thread(target=increase_clock_every_second, args=(pipeline_context,))
    clock_thread.start()

    prev_clock_idx = -1
    for tasks in pipeline_context.get_schedule():
        assert isinstance(tasks, Task)
        assert pipeline_context.clock_idx == prev_clock_idx + 1
        prev_clock_idx = pipeline_context.clock_idx

    assert pipeline_context.clock_idx == TOTAL_SCHEDULES


def test_get_syncronous_schedule():
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1
    WORLD_SIZE = TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE * DATA_PARALLEL_SIZE

    spawn(
        run_get_syncronous_schedule,
        world_size=WORLD_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
    )
