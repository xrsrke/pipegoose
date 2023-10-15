import threading
from enum import Enum, auto
from typing import List

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel2._comm import set_pipeline_context
from pipegoose.nn.pipeline_parallel2._utils import get_partition_idx, is_last_stage
from pipegoose.nn.pipeline_parallel2.scheduler import BaseScheduler


class TrainingState(Enum):
    """Training state of the pipeline engine."""

    IDLE = auto()
    FORWARD = auto()
    BACKWARD = auto()
    FINISHED = auto()


class PipelineContext:
    """A context that holds information about the pipeline execution."""

    def __init__(self, scheduler: BaseScheduler, parallel_context: ParallelContext):
        self.scheduler = scheduler
        self.parallel_context = parallel_context

        self._clock_idx: int = 0
        self._state: TrainingState = TrainingState.IDLE
        # NOTE: block CPU thread until the next clock cycle
        self._wait_new_clock_cycle = threading.Condition()

        set_pipeline_context(pipeline_context=self)

    @property
    def state(self) -> TrainingState:
        return self._state

    def forward(self):
        self._state = TrainingState.FORWARD

    def backward(self):
        self._state = TrainingState.BACKWARD

    def finish(self):
        self._state = TrainingState.FINISHED

    @property
    def partition_idx(self) -> int:
        parallel_context = self.parallel_context
        pipeline_stage_idx = get_partition_idx(parallel_context)
        return pipeline_stage_idx

    @property
    def clock_idx(self) -> int:
        """Get the current clock cycle in the pipeline execution."""
        return self._clock_idx

    def increase_a_clock_cycle(self):
        """Increase the current clock cycle in the pipline by 1."""
        # TODO: add assert maximum clock cycles
        with self._wait_new_clock_cycle:
            self._clock_idx += 1
            self._wait_new_clock_cycle.notify_all()

    @property
    def schedule(self) -> List:
        """Get the current schedule of this partition."""
        return self._get_schedule_from_partition(self.clock_idx, self.partition_idx, training_state=self.state)

    @property
    def schedules(self) -> List:
        """Get the schedule for entire training run."""
        return self.scheduler.get_schedules()

    def _get_schedule_from_training_state(self, training_state: TrainingState) -> List:
        """Get the schedule from a given training state."""
        STATE_TO_SCHEDULES = {
            training_state.FORWARD: self.scheduler.get_forward_schedules,
            training_state.BACKWARD: self.scheduler.get_backward_schedules,
        }
        return STATE_TO_SCHEDULES[training_state]()

    def get_schedule(self):
        with self._wait_new_clock_cycle:
            while self.clock_idx < self.scheduler.total_clock_cycles:
                schedules = self._get_schedule_from_partition(self.clock_idx, self.partition_idx, training_state=self.state)
                yield schedules

                # NOTE: wait for the next clock cycle
                print(
                    f"waiting for the next clock cycle, clock_idx={self.clock_idx}, rank={self.parallel_context.get_local_rank(ParallelMode.GLOBAL)}"
                )
                self._wait_new_clock_cycle.wait()

    def _get_schedule_from_partition(self, clock_idx: int, partition_idx: int, training_state: TrainingState):
        """Get the schedule of a partition at a certain clock cycle."""

        assert clock_idx >= 0, "Clock cycle index must be greater than or equal to 0."
        assert partition_idx >= 0, "Partition index must be greater than or equal to 0."

        schedules = self._get_schedule_from_training_state(training_state)[clock_idx]
        schedule_of_this_partition = [schedule for schedule in schedules if schedule.partition_idx == partition_idx]

        return schedule_of_this_partition

    def _get_schedule_from_microbatch(self, clock_idx: int, microbatch_idx: int, training_state: TrainingState):
        """Get the schedule of a microbatch at a certain clock cycle."""
        assert clock_idx >= 0, "Clock cycle index must be greater than or equal to 0."
        assert microbatch_idx >= 0, "Microbatch index must be greater than or equal to 0."

        schedules = self._get_schedule_from_training_state(training_state)[clock_idx]
        schedule_of_this_microbatch = [schedule for schedule in schedules if schedule.microbatch_idx == microbatch_idx]

        return schedule_of_this_microbatch

    def get_next_schedule_from_microbatch(self, microbatch_idx: int):
        """Get the schedule of a micro-batch in the next clock cycle."""
        next_clock_idx = self.clock_idx + 1
        schedule = self._get_schedule_from_microbatch(
            clock_idx=next_clock_idx, microbatch_idx=microbatch_idx, training_state=self.state
        )
        return schedule

    @property
    def is_first_stage(self) -> bool:
        return self.partition_idx == 0

    @property
    def is_last_stage(self) -> bool:
        return is_last_stage(self.parallel_context)
