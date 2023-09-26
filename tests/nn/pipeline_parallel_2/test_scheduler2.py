from pipegoose.nn.pipeline_parallel2._job.job_type import JobType
from pipegoose.nn.pipeline_parallel2.scheduler import GPipeScheduler


def test_generate_schedules_using_gpipe_scheduler():
    N_MICROBATCHES = 4
    N_PARTITIONS = 3
    JOB_TYPES = [JobType.FORWARD, JobType.BACKWARD]

    TOTAL_CLOCK_CYCLES_IN_FORWARD = N_MICROBATCHES + N_PARTITIONS - 1
    TOTAL_CLOCK_CYCLES = TOTAL_CLOCK_CYCLES_IN_FORWARD * 2

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
