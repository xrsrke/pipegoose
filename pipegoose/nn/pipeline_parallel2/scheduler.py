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

    @abstractclassmethod
    def get_forward_schedules(self):
        """Return the forward schedule for the whole training run."""
        raise NotImplementedError

    @abstractclassmethod
    def get_backward_schedules(self):
        """Return the backward schedule for the whole training run."""
        raise NotImplementedError

    @abstractclassmethod
    def total_clock_cycles(self):
        """Return the total number of clock cycles."""
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
        forward_schedules = self.get_forward_schedules()
        backward_schedules = self.get_backward_schedules()

        # NOTE: combine forward and backward schedule into a full schedule
        forward_schedules.extend(backward_schedules)

        return forward_schedules

    def get_forward_schedules(self) -> List[List[Task]]:
        schedules = []
        n_clock_cycles = self.n_partitions + self.n_microbatches - 1
        for clock_idx in range(n_clock_cycles):
            start_partrition = max(clock_idx + 1 - self.n_microbatches, 0)
            end_partition = min(clock_idx + 1, self.n_partitions)

            tasks = []
            for partition_idx in range(start_partrition, end_partition):
                microbatch_idx = clock_idx - partition_idx
                task = Task(JobType.FORWARD, microbatch_idx, partition_idx)
                tasks.append(task)

            schedules.append(tasks)
        return schedules

    def get_backward_schedules(self) -> List[List[Task]]:
        from copy import deepcopy

        forward_schedules = self.get_forward_schedules()
        n_clock_cycles = len(forward_schedules)
        backward_schedules = deepcopy(forward_schedules)
        backward_schedules.reverse()

        for clock_idx in range(n_clock_cycles):
            for task in backward_schedules[clock_idx]:
                task.job_type = JobType.BACKWARD

        return backward_schedules

    @property
    def total_clock_cycles(self) -> int:
        return len(self.get_schedules())

    @property
    def total_forward_clock_cycles(self) -> int:
        """Return the total number of clock cycles required to run the forward pass."""
        return len(self.get_forward_schedules())

    @property
    def total_backward_clock_cycles(self) -> int:
        """Return the total number of clock cycles required to run the forward pass."""
        return len(self.get_backward_schedules())


def get_scheduler(scheduler_type: SchedulerType) -> BaseScheduler:
    scheduler_type_to_scheduler = {
        SchedulerType.GPIPE: GPipeScheduler,
    }

    return scheduler_type_to_scheduler[scheduler_type]
