from abc import ABC, abstractclassmethod
from typing import List
from dataclasses import dataclass

from pipegoose.nn.pipeline_parallel2._job.job_type import JobType


@dataclass
class Task:
    job_type: JobType
    microbatch_idx: int
    partition_idx: int


class BaseScheduler(ABC):
    @abstractclassmethod
    def get_schedules(self):
        """Return the schedule for the whole training run."""
        raise NotImplementedError

    @abstractclassmethod
    def clock_idx(self) -> int:
        """Return the current clock cycle index."""
        raise NotImplementedError

    @abstractclassmethod
    def start(self):
        pass

    @abstractclassmethod
    def is_running(self):
        pass


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

        self._clock_idx = None
        self._schedules = None

    @property
    def clock_idx(self):
        pass

    @property
    def is_running(self):
        pass

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

    def start(self):
        self._schedules = self.get_schedules()


class JobTracker:
    def __init__(self, n_microbatches: int, n_partitions: int):
        self.n_microbatches = n_microbatches
        self.n_partitions = n_partitions

        self._progress = {
            partition_idx: {microbatch_idx: False for microbatch_idx in range(n_microbatches)}
            for partition_idx in range(n_partitions)
        }
