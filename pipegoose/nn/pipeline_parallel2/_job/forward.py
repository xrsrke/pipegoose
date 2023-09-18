from copy import deepcopy
from typing import Tuple

import torch

from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel2._comm import send_package
from pipegoose.nn.pipeline_parallel2._job.callback import Callback
from pipegoose.nn.pipeline_parallel2._job.job import Job
from pipegoose.nn.pipeline_parallel2._package import Package


def get_activation_name(microbatch_idx: int, partition_idx: int) -> Tuple[int, int]:
    return (microbatch_idx, partition_idx)


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
        from pipegoose.nn.pipeline_parallel2.queue import save_activations

        is_training = self.job.input.metadata.training.is_training
        if is_training is True:
            # TODO: refactor
            microbatch_idx = self.job.input.metadata.microbatch_idx
            partition_idx = self.job.input.metadata.partition_idx

            key = get_activation_name(microbatch_idx, partition_idx)
            save_activations(key, self.job.output.data)


class SendForwardPackageCallback(Callback):
    """Send the output of a forward job to the next pipeline stage."""

    order = 5

    def after_compute(self):
        parallel_context = self.job.pipeline_context.parallel_context

        if parallel_context.get_world_size(ParallelMode.GLOBAL) > 1 and parallel_context.pipeline_parallel_size > 1:
            output = self.job.output
            assert isinstance(output, Package), f"output must be an instance of Package, got {type(output)}"
            send_package(output, parallel_context)
