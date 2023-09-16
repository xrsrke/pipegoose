import pytest
from torch import nn

from pipegoose.nn.pipeline_parallel2.pipeline_context import PipelineContext
from pipegoose.nn.pipeline_parallel2.scheduler import SchedulerType, get_scheduler
from pipegoose.testing.utils import init_parallel_context, spawn


def run_pipeline_context(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    N_PARTITIONS = 5
    N_MICROBATCHES = 4

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    partitions = [nn.Linear(10, 10) for _ in range(N_PARTITIONS)]
    scheduler = get_scheduler(SchedulerType.GPIPE)(N_MICROBATCHES, N_PARTITIONS)

    pipeline_context = PipelineContext(partitions, scheduler, parallel_context)

    assert isinstance(pipeline_context.partition_idx, int)
    assert isinstance(pipeline_context.get_partition_forward(), nn.Module)

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
