from copy import deepcopy
from typing import Any

import torch

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel2._job.callback import Callback
from pipegoose.nn.pipeline_parallel2._job.job import Job
from pipegoose.nn.pipeline_parallel2._package import Package
from pipegoose.nn.pipeline_parallel2.exception import PipelineGradientFlowError
from pipegoose.nn.pipeline_parallel2.pipeline_context import PipelineContext
from pipegoose.nn.pipeline_parallel2.queue import (
    get_input_activations,
    get_output_activations,
)


class _SaveGradLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, key, metadata, tensor: torch.Tensor):
        print("forward of saving grad loss", key)
        ctx.key = key
        ctx.package_metadata = metadata
        new_tensor = tensor.detach().clone()
        return new_tensor

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        with torch.autograd.set_grad_enabled(False):
            from pipegoose.nn.pipeline_parallel2.queue import (
                _SAVED_GRAD_LOSS,
                _SAVED_METADATA_of_GRAD_LOSS,
            )

            key = ctx.key
            print("backward of saving grad loss", ctx.key)

            _SAVED_GRAD_LOSS[key] = grad_output
            _SAVED_METADATA_of_GRAD_LOSS[key] = ctx.package_metadata
            # NOTE: prevent the grad from flowing back to the input of the pipeline stage
            # since we only do one backward pass at a time
        return (None, None, None)


def save_grad_loss(package: Package) -> Package:
    key = (package.metadata.microbatch_idx, package.metadata.partition_idx)
    package.data = _SaveGradLossFunction.apply(key, package.metadata, package.data)
    if package.metadata.partition_idx == 3:
        assert 1 == 1
    return package


class CreateBackwardOutputPackageCallback(Callback):
    """Create a new package for the output of a backward job."""

    order = 0

    def __init__(self, parallel_context: ParallelContext, pipeline_context: PipelineContext):
        self.parallel_context = parallel_context
        self.pipeline_context = pipeline_context

    def after_compute(self):
        data = self.job.output
        orig_metadata = deepcopy(self.job.input.metadata)

        package = Package(data, orig_metadata)

        if not self.pipeline_context.is_first_stage:
            package = self._update_next_pipeline_stage(package)
            package = self._update_src_and_dst_rank(package)

        self.job.output = package

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
        package.metadata.dst = self.parallel_context.get_prev_global_rank(ParallelMode.PIPELINE)

        return package


class SendBackwardPackageCallback(Callback):
    """Send the gradients of a backward job to a previous pipeline stage."""

    order = 1

    def __init__(self, parallel_context: ParallelContext):
        self.parallel_context = parallel_context

    def after_compute(self):
        if self.parallel_context.pipeline_parallel_size > 1:
            from pipegoose.nn.pipeline_parallel2._comm import send_package

            output = self.job.output
            assert isinstance(output, Package), f"output must be an instance of Package, got {type(output)}"
            send_package(output, self.parallel_context)


class BackwardJob(Job):
    """Do backward pass."""

    def __init__(self, *args, is_scheduled: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_scheduled: bool = is_scheduled

    def run_compute(self) -> torch.Tensor:
        microbatch_idx = self.input.metadata.microbatch_idx
        partition_idx = self.input.metadata.partition_idx
        prev_grad = self.input.data

        from pipegoose.nn.pipeline_parallel2._comm import get_pipeline_context

        pipeline_context = get_pipeline_context()
        rank = pipeline_context.parallel_context.get_global_rank()

        input = get_input_activations(microbatch_idx, partition_idx)
        output = get_output_activations(microbatch_idx, partition_idx, self.is_scheduled)

        if input.requires_grad is False and partition_idx != 0:
            raise PipelineGradientFlowError(
                f"Please set .requires_grad = True to input activations. Gradients can't flow back to the input of the pipeline stage, rank={rank}, microbatch_idx={microbatch_idx}, partition_idx={partition_idx}"
            )

        def is_last_microbatch(microbatch_idx):
            return microbatch_idx == pipeline_context.num_microbatches - 1

        # if not is_last_microbatch(microbatch_idx):
        #     output = output.detach().requires_grad_(True)

        if rank == 3 and microbatch_idx == 4:
            assert 1 == 1

        # new_output = output.detach().requires_grad_(True)
        torch.autograd.backward(output, grad_tensors=prev_grad)

        if partition_idx == 0:
            return

        if input.grad is None:
            raise PipelineGradientFlowError(
                "Gradients can't flow back to the input of the pipeline stage, rank={rank}, microbatch_idx={microbatch_idx}, partition_idx={partition_idx}"
            )

        # # TODO: remove this, since the grads is stored in module's weights
        # # and we do gradient accumulation, we don't need return grads or send to other stages
        # assert isinstance(input.grad, torch.Tensor)

        # print(f"executing backward job, rank={rank}, microbatch_idx={microbatch_idx}, partition_idx={partition_idx}")
        print(
            f"rank={rank}, microbatch_idx={microbatch_idx}, partition_idx={partition_idx}, yay! gradients: {input.grad.shape}"
        )

        return input.grad
