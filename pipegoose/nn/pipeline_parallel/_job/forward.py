from copy import deepcopy

import torch

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel._job.callback import Callback
from pipegoose.nn.pipeline_parallel._job.job import Job
from pipegoose.nn.pipeline_parallel._package import Package
from pipegoose.nn.pipeline_parallel.pipeline_context import PipelineContext
from pipegoose.nn.pipeline_parallel.sync.handshake import get_progress_tracker


class ForwardJob(Job):
    def run_compute(self) -> torch.Tensor:
        is_training = self.input.metadata.training.is_training
        with torch.set_grad_enabled(is_training):
            output = self.function(self.input.data)
        return output


class CreateForwardOutputPackageCallback(Callback):
    """Create a new package for the output of a forward job."""

    order = 0

    def __init__(self, parallel_context: ParallelContext, pipeline_context: PipelineContext):
        self.parallel_context = parallel_context
        self.pipeline_context = pipeline_context

    def after_compute(self):
        self._save_buffer_for_backward()

        input_metadata = deepcopy(self.job.input.metadata)
        package = Package(self.job.output, input_metadata)

        if not self.pipeline_context.is_last_stage:
            package = self._update_next_pipeline_stage(package)
            package = self._update_src_and_dst_rank(package)

        self.job.output = package

    def _save_buffer_for_backward(self):
        from pipegoose.nn.pipeline_parallel.queue import (
            save_input_activations,
            save_output_activations,
        )

        microbatch_idx = self.job.input.metadata.microbatch_idx
        partition_idx = self.job.input.metadata.partition_idx
        save_input_activations(self.job.input.data, microbatch_idx, partition_idx)
        save_output_activations(self.job.output, microbatch_idx, partition_idx)

    def _update_next_pipeline_stage(self, package: Package) -> Package:
        microbatch_idx = package.metadata.microbatch_idx

        # NOTE: determine which pipeline stage to send the output to
        next_schedule = self.pipeline_context.get_next_schedule_from_microbatch(microbatch_idx)

        # NOTE: because currently each pipeline stage only has one task at a time
        # so we hardcode and select that one
        # TODO: take into account that a pipeline stage can has more than one task
        # in a clock cycle, then find the correspond task to send the output to
        next_partition = next_schedule[0].partition_idx
        package.metadata.partition_idx = next_partition

        return package

    def _update_src_and_dst_rank(self, package: Package) -> Package:
        package.metadata.src = self.parallel_context.get_global_rank()
        package.metadata.dst = self.parallel_context.get_next_global_rank(ParallelMode.PIPELINE)

        return package


class SaveBufferForBackwardCallback(Callback):
    order = 1

    def ater_compute(self):
        # from pipegoose.nn.pipeline_parallel.queue import save_input_activations, save_output_activations

        # microbatch_idx = self.job.input.metadata.microbatch_idx
        # partition_idx = self.job.input.metadata.partition_idx
        # save_input_activations(self.job.input.data, microbatch_idx, partition_idx)
        # save_output_activations(self.job.output.data, microbatch_idx, partition_idx)
        pass


class SendForwardPackageCallback(Callback):
    """Send the output of a forward job to the next pipeline stage."""

    order = 5

    def __init__(self, parallel_context: ParallelContext):
        self.parallel_context = parallel_context

    def after_compute(self):
        from pipegoose.nn.pipeline_parallel._comm import send_package

        if self.parallel_context.pipeline_parallel_size > 1:
            output = self.job.output
            assert isinstance(output, Package), f"output must be an instance of Package, got {type(output)}"
            send_package(output, self.parallel_context)


class ConfirmCompleteATaskToProgressTracker(Callback):
    """Confirm that a task is completed to progress tracker."""

    order = 6

    def __init__(self, parallel_context: ParallelContext):
        assert get_progress_tracker() is not None, "Progress tracker must be initialized before using this callback"

        world_size = parallel_context.get_world_size(ParallelMode.GLOBAL)
        assert world_size > 1, "Progress tracker is only used in distributed training"

    def after_compute(self):
        microbatch_idx = self.job.input.metadata.microbatch_idx
        partition_idx = self.job.input.metadata.partition_idx
        key = (microbatch_idx, partition_idx)

        progress_tracker = get_progress_tracker()
        progress_tracker.confirm(key)
