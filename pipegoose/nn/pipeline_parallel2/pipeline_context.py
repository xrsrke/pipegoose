from typing import List

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel2.scheduler import BaseScheduler


class PipelineContext:
    """A context that holds information about the pipeline execution."""

    def __init__(self, scheduler: BaseScheduler, parallel_context: ParallelContext):
        self.scheduler = scheduler
        self.parallel_context = parallel_context

        self._clock_idx = 0

    @property
    def partition_idx(self) -> int:
        parallel_context = self.parallel_context
        rank = parallel_context.get_local_rank(ParallelMode.PIPELINE)
        n_ranks_per_group = len(parallel_context.get_ranks_in_group(ParallelMode.PIPELINE))
        pipeline_stage_idx = rank // n_ranks_per_group
        return pipeline_stage_idx

    @property
    def clock_idx(self) -> int:
        """Get the current clock cycle in the pipeline execution."""
        return self._clock_idx

    def increase_a_clock_cycle(self):
        """Increase the current clock cycle in the pipline by 1."""
        # TODO: add assert maximum clock cycles
        self._clock_idx += 1

    @property
    def schedule(self) -> List:
        """Get the current schedule of this partition."""
        return self.get_schedule_from_partition(self.clock_idx, self.partition_idx)

    @property
    def schedules(self) -> List:
        """Get the schedule for entire training run."""
        return self.scheduler.get_schedules()

    def get_schedule_from_partition(self, clock_idx: int, partition_idx: int):
        """Get the schedule of a partition at a certain clock cycle."""
        assert clock_idx >= 0, "Clock cycle index must be greater than or equal to 0."
        assert partition_idx >= 0, "Partition index must be greater than or equal to 0."

        schedules = self.schedules[clock_idx]
        schedule_of_this_partition = [schedule for schedule in schedules if schedule.partition_idx == partition_idx]

        return schedule_of_this_partition

    def get_schedule_from_microbatch(self, clock_idx: int, microbatch_idx: int):
        """Get the schedule of a microbatch at a certain clock cycle."""
        assert clock_idx >= 0, "Clock cycle index must be greater than or equal to 0."
        assert microbatch_idx >= 0, "Microbatch index must be greater than or equal to 0."

        schedules = self.schedules[clock_idx]
        schedule_of_this_microbatch = [schedule for schedule in schedules if schedule.microbatch_idx == microbatch_idx]

        return schedule_of_this_microbatch

    def get_next_schedule_from_microbatch(self, microbatch_idx):
        """Get the schedule of a micro-batch in the next clock cycle."""
        next_clock_idx = self.clock_idx + 1
        schedule = self.get_schedule_from_microbatch(
            clock_idx=next_clock_idx,
            microbatch_idx=microbatch_idx,
        )
        return schedule
