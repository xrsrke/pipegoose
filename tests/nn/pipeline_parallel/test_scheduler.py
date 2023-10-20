from pipegoose.nn.pipeline_parallel._job.job_type import JobType
from pipegoose.nn.pipeline_parallel.scheduler import GPipeScheduler

N_MICROBATCHES = 4
N_PARTITIONS = 3


def test_generate_forward_and_backward_schedules_using_gpipe_scheduler():
    TOTAL_CLOCK_CYCLES_IN_FORWARD = N_MICROBATCHES + N_PARTITIONS - 1
    TOTAL_CLOCK_CYCLES = TOTAL_CLOCK_CYCLES_IN_FORWARD * 2
    JOB_TYPES = [JobType.FORWARD, JobType.BACKWARD]

    scheduler = GPipeScheduler(N_MICROBATCHES, N_PARTITIONS)

    assert scheduler.total_clock_cycles == TOTAL_CLOCK_CYCLES

    schedules = scheduler.get_schedules()
    assert len(schedules) == TOTAL_CLOCK_CYCLES

    for tasks in schedules:
        assert isinstance(tasks, list)

        for task in tasks:
            assert task.job_type in JOB_TYPES
            assert isinstance(task.partition_idx, int)
            assert isinstance(task.microbatch_idx, int)


def test_generate_forward_schedules_using_gpipe_scheduler():
    TOTAL_CLOCK_CYCLES_IN_FORWARD = N_MICROBATCHES + N_PARTITIONS - 1

    scheduler = GPipeScheduler(N_MICROBATCHES, N_PARTITIONS)
    schedules = scheduler.get_forward_schedules()

    assert len(schedules) == TOTAL_CLOCK_CYCLES_IN_FORWARD

    for tasks in schedules:
        assert isinstance(tasks, list)

        for task in tasks:
            assert task.job_type == JobType.FORWARD
            assert isinstance(task.partition_idx, int)
            assert isinstance(task.microbatch_idx, int)


def test_generate_backward_schedules_using_gpipe_scheduler():
    TOTAL_CLOCK_CYCLES_IN_BACKWARD = N_MICROBATCHES + N_PARTITIONS - 1

    scheduler = GPipeScheduler(N_MICROBATCHES, N_PARTITIONS)
    schedules = scheduler.get_backward_schedules()

    assert len(schedules) == TOTAL_CLOCK_CYCLES_IN_BACKWARD

    for tasks in schedules:
        assert isinstance(tasks, list)

        for task in tasks:
            assert task.job_type == JobType.BACKWARD
            assert isinstance(task.partition_idx, int)
            assert isinstance(task.microbatch_idx, int)
