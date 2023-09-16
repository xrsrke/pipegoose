import pytest

from pipegoose.nn.pipeline_parallel2._job.job_type import JobType
from pipegoose.nn.pipeline_parallel2.scheduler import Scheduler, get_scheduler
from pipegoose.testing.utils import spawn


def test_generate_schedules_using_gpipe_scheduler():
    N_MICROBATCHES = 4
    N_PARTITIONS = 3
    JOB_TYPES = [JobType.FORWARD, JobType.BACKWARD]

    TOTAL_CLOCK_CYCLES_IN_FORWARD = N_MICROBATCHES + N_PARTITIONS - 1
    TOTAL_CLOCK_CYCLES = TOTAL_CLOCK_CYCLES_IN_FORWARD * 2

    scheduler = get_scheduler(Scheduler.GPIPE)
    schedules = scheduler(N_MICROBATCHES, N_PARTITIONS).get_schedules()

    # schedules = GPipeScheduler(N_MICROBATCHES, N_PARTITIONS).get_schedules()

    assert len(schedules) == TOTAL_CLOCK_CYCLES

    for tasks in schedules:
        assert isinstance(tasks, list)

        for task in tasks:
            assert task.job_type in JOB_TYPES
            assert isinstance(task.partition_idx, int)
            assert isinstance(task.microbatch_idx, int)


def run_syncronous_gpipe_scheduler(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    # parallel_context = init_parallel_context(
    #     rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    # )

    # local_clock_idx = 0

    # scheduler = GPipeScheduler(parallel_context)
    # scheduler.start()

    # assert scheduler.clock_idx == local_clock_idx

    # for _ in range(5):
    #     # NOTE: simulate that different nodes have different processing times
    #     sleep(random.uniform(1, 5))

    #     scheduler.confirm()

    #     assert scheduler.clock_idx == local_clock_idx + 1
    #     local_clock_idx += 1
    pass


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
