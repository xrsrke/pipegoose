import random
from time import sleep

import pytest

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.pipeline_parallel2.scheduler import GPipeScheduler
from pipegoose.nn.pipeline_parallel2._job.job_type import JobType
from pipegoose.testing.utils import spawn


def init_parallel_context(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    parallel_context = ParallelContext(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        host="localhost",
        port=port,
        seed=69,
        backend="gloo",
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )

    return parallel_context


def test_generate_schedule_using_gpipe_scheduler():
    N_MICROBATCHES = 4
    N_PARTITIONS = 3
    JOB_TYPES = [JobType.FORWARD, JobType.BACKWARD]

    schedules = GPipeScheduler.get_schedule(N_MICROBATCHES, N_PARTITIONS)

    for tasks in schedules:
        assert isinstance(tasks, list)
        for task in tasks:
            assert task.job_type in JOB_TYPES
            assert isinstance(task.partition_idx, int)
            assert isinstance(task.microbatch_idx, int)


def run_syncronous_gpipe_scheduler(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    local_clock_idx = 0

    scheduler = GPipeScheduler(parallel_context)
    scheduler.start()

    assert scheduler.clock_idx == local_clock_idx

    for _ in range(5):
        # NOTE: simulate that different nodes have different processing times
        sleep(random.uniform(1, 5))

        scheduler.confirm()

        assert scheduler.clock_idx == local_clock_idx + 1
        local_clock_idx += 1


@pytest.mark.skip
@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
def test_syncronous_scheduler(pipeline_parallel_size):
    TENSOR_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    spawn(
        run_syncronous_gpipe_scheduler,
        world_size=pipeline_parallel_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=DATA_PARALLEL_SIZE,
    )
