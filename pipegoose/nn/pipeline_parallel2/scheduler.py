from abc import ABC, abstractclassmethod
from enum import Enum, auto
from typing import List

from pipegoose.nn.pipeline_parallel2._job.job_type import JobType
from pipegoose.nn.pipeline_parallel2.task import Task


class SchedulerType(Enum):
    GPIPE = auto()


class BaseScheduler(ABC):
    @abstractclassmethod
    def get_schedules(self):
        """Return the schedule for the whole training run."""
        raise NotImplementedError


class GPipeScheduler(BaseScheduler):
    """
    torchgpipe: On-the-fly Pipeline Parallelism for Training Giant Models
    https://arxiv.org/abs/2004.09910

    Section 3.2.1: Forward Dependency: Deterministic Clock-cycle
    """

    def __init__(self, n_microbatches: int, n_partitions: int):
        assert (
            n_microbatches > 0
        ), "The number of microbatches must be \
            greater than 0"
        assert (
            n_partitions > 0
        ), "The number of partitions must be \
            greater than 0"

        self.n_microbatches = n_microbatches
        self.n_partitions = n_partitions

    def get_schedules(self) -> List[List[Task]]:
        def generate_forward_schedule(n_microbatches, n_partitions):
            schedules = []
            n_clock_cycles = n_partitions + n_microbatches - 1
            for clock_idx in range(n_clock_cycles):
                start_partrition = max(clock_idx + 1 - n_microbatches, 0)
                end_partition = min(clock_idx + 1, n_partitions)

                tasks = []
                for partition_idx in range(start_partrition, end_partition):
                    microbatch_idx = clock_idx - partition_idx
                    task = Task(JobType.FORWARD, microbatch_idx, partition_idx)
                    tasks.append(task)

                schedules.append(tasks)
            return schedules

        def generate_backward_schedule(forward_schedule):
            from copy import deepcopy

            n_clock_cycles = len(forward_schedule)
            backward_schedule = deepcopy(forward_schedule)
            backward_schedule.reverse()

            for clock_idx in range(n_clock_cycles):
                for task in backward_schedule[clock_idx]:
                    task.job_type = JobType.BACKWARD

            return backward_schedule

        forward_schedule = generate_forward_schedule(self.n_microbatches, self.n_partitions)
        backward_schedule = generate_backward_schedule(forward_schedule)

        # NOTE: combine forward and backward schedule into a full schedule
        forward_schedule.extend(backward_schedule)

        return forward_schedule

    @property
    def total_clock_cycles(self) -> int:
        return len(self.get_schedules())


def get_scheduler(scheduler_type: SchedulerType) -> BaseScheduler:
    scheduler_type_to_scheduler = {
        SchedulerType.GPIPE: GPipeScheduler,
    }

    return scheduler_type_to_scheduler[scheduler_type]
