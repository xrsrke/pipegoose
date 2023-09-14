from typing import List

from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.pipeline_parallel2.scheduler import BaseScheduler


class PipelineContext:
    """A context that holds information about the current pipeline execution."""

    def __init__(self, partitions: List[nn.Module], scheduler: BaseScheduler, parallel_context: ParallelContext):
        self.partitions = partitions
        self.scheduler = scheduler
        self.parallel_context = parallel_context

        self._clock_idx = 0

    @property
    def partition_idx(self) -> int:
        return 0

    def get_partition_forward(self) -> nn.Module:
        return self.partitions[self.partition_idx]

    @property
    def current_clock_idx(self) -> int:
        return self._clock_idx

    @property
    def current_schedule(self) -> List:
        return self.scheduler.get_schedules()[self.current_clock_idx]

    @property
    def schedules(self) -> List:
        return self.scheduler.get_schedules()
