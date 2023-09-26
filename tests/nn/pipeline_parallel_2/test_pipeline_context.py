import threading

import pytest

from pipegoose.nn.pipeline_parallel2.pipeline_context import PipelineContext
from pipegoose.nn.pipeline_parallel2.scheduler import SchedulerType, get_scheduler
from pipegoose.nn.pipeline_parallel2.task import Task
from pipegoose.testing.utils import init_parallel_context, spawn


def run_pipeline_context(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    N_PARTITIONS = 5
    N_MICROBATCHES = 4

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    scheduler = get_scheduler(SchedulerType.GPIPE)(N_MICROBATCHES, N_PARTITIONS)

    pipeline_context = PipelineContext(scheduler, parallel_context)

    assert isinstance(pipeline_context.partition_idx, int)

    assert pipeline_context.clock_idx == 0
    assert isinstance(pipeline_context.schedule, list)
    assert isinstance(pipeline_context.schedules, list)
    assert isinstance(pipeline_context.get_schedule_from_partition(clock_idx=3, partition_idx=2), list)
    assert isinstance(pipeline_context.get_schedule_from_microbatch(clock_idx=3, microbatch_idx=0), list)

    next_schedules = pipeline_context.get_next_schedule_from_microbatch(microbatch_idx=0)
    assert isinstance(next_schedules, list)

    for task in next_schedules:
        # NOTE: expect all the tasks that being processed in the current clock cycle
        # to be send to the next partition in the next clock cycle
        assert task.partition_idx == pipeline_context.partition_idx + 1

    CURRENT_CLOCK_IDX = pipeline_context.clock_idx
    pipeline_context.increase_a_clock_cycle()

    assert pipeline_context.clock_idx == CURRENT_CLOCK_IDX + 1


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
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
            # time.sleep(1)
            pipeline_context.increase_a_clock_cycle()

    pipeline_context = PipelineContext(scheduler, parallel_context)
    clock_thread = threading.Thread(target=increase_clock_every_second, args=(pipeline_context,))
    clock_thread.start()

    prev_clock_idx = -1
    for tasks in pipeline_context.get_schedule():
        assert isinstance(tasks, Task)
        assert pipeline_context.clock_idx == prev_clock_idx + 1
        prev_clock_idx = pipeline_context.clock_idx

    assert pipeline_context.clock_idx == TOTAL_SCHEDULES


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
def test_get_syncronous_schedule(pipeline_parallel_size):
    TENSOR_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    world_size = pipeline_parallel_size * TENSOR_PARALLEL_SIZE * DATA_PARALLEL_SIZE

    spawn(
        run_get_syncronous_schedule,
        world_size=world_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=DATA_PARALLEL_SIZE,
    )
