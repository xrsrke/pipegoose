from copy import deepcopy

import torch

from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel2._comm import send_package
from pipegoose.nn.pipeline_parallel2._job.callback import Callback
from pipegoose.nn.pipeline_parallel2._job.job import Job
from pipegoose.nn.pipeline_parallel2._package import Package
from pipegoose.nn.pipeline_parallel2.sync.handshake import get_progress_tracker


class ForwardJob(Job):
    def run_compute(self) -> torch.Tensor:
        return self.function(self.input.data)


class CreateForwardOutputPackageCallback(Callback):
    """Create a new package for the output of a forward job."""

    order = 0

    def after_compute(self):
        data = self.job.output
        input_metadata = deepcopy(self.job.input.metadata)

        package = Package(data, input_metadata)

        if not self.job.pipeline_context.is_last_stage:
            package = self._update_next_pipeline_stage(package)
            package = self._update_src_and_dst_rank(package)

        self.job.output = package

    def _update_next_pipeline_stage(self, package: Package) -> Package:
        pipeline_context = self.job.pipeline_context
        microbatch_idx = package.metadata.microbatch_idx

        # NOTE: determine which pipeline stage to send the output to
        next_schedule = pipeline_context.get_next_schedule_from_microbatch(microbatch_idx)

        # NOTE: because currently each pipeline stage only has one task at a time
        # so we hardcode and select that one
        # TODO: take into account that a pipeline stage can has more than one task
        # in a clock cycle, then find the correspond task to send the output to

        # print("---------- _update_next_pipeline_stage -----------")
        # print(f"rank: {self.job.pipeline_context.parallel_context.get_local_rank(ParallelMode.GLOBAL)}")
        # print(f"clock_idx: {self.job.pipeline_context.clock_idx}")
        # print(
        #     f"schedules = {self.job.pipeline_context.get_schedule_from_microbatch(clock_idx=self.job.pipeline_context.clock_idx+1, microbatch_idx=microbatch_idx)}"
        # )
        # print(f"microbatch_idx: {microbatch_idx}")
        # print(f"next_schedule: {next_schedule}")

        next_partition = next_schedule[0].partition_idx
        package.metadata.partition_idx = next_partition

        return package

    def _update_src_and_dst_rank(self, package: Package) -> Package:
        pipeline_context = self.job.pipeline_context
        parallel_context = pipeline_context.parallel_context

        package.metadata.src = parallel_context.get_global_rank()
        package.metadata.dst = parallel_context.get_next_global_rank(ParallelMode.PIPELINE)

        return package


class SaveActivationIfTrainingCallback(Callback):
    """Save the activation of a forward job for backward pass if training."""

    order = 2

    def after_compute(self):
        is_training = self.job.input.metadata.training.is_training
        if is_training is True:
            from pipegoose.nn.pipeline_parallel2.queue import SavedActivation

            # TODO: refactor
            microbatch_idx = self.job.input.metadata.microbatch_idx
            partition_idx = self.job.input.metadata.partition_idx

            key = SavedActivation.get_key(microbatch_idx, partition_idx)
            SavedActivation.save_activations(key, self.job.output.data)


class SendForwardPackageCallback(Callback):
    """Send the output of a forward job to the next pipeline stage."""

    order = 5

    def after_compute(self):
        parallel_context = self.job.pipeline_context.parallel_context

        if parallel_context.pipeline_parallel_size > 1:
            output = self.job.output
            assert isinstance(output, Package), f"output must be an instance of Package, got {type(output)}"
            send_package(output, parallel_context)


class ConfirmCompleteATaskToProgressTracker(Callback):
    """Confirm that a task is completed to progress tracker."""

    order = 6

    def after_compute(self):
        progress_tracker = get_progress_tracker()
        microbatch_idx = self.job.input.metadata.microbatch_idx
        partition_idx = self.job.input.metadata.partition_idx
        # task = Task(
        #     job_type=JobType.FORWARD,
        #     microbatch_idx=microbatch_idx,
        #     partition_idx=self.job.input.metadata.partition_idx,
        # )
        key = (microbatch_idx, partition_idx)
        progress_tracker.confirm(key)
