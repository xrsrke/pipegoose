import torch

from pipegoose.nn.pipeline_parallel2._job.callback import Callback
from pipegoose.nn.pipeline_parallel2._job.job import Job
from pipegoose.nn.pipeline_parallel2._package import Package
from pipegoose.nn.pipeline_parallel2.exception import PipelineGradientFlowError
from pipegoose.nn.pipeline_parallel2.queue import (
    get_input_activations,
    get_output_activations,
)


class CreateBackwardOutputPackageCallback(Callback):
    """Create a new package for the output of a backward job."""

    def after_compute(self):
        data = self.job.output
        orig_metadata = self.job.input.metadata

        package = Package(data, orig_metadata)
        package.metadata.partition_idx -= 1

        self.job.output = package


class SendBackwardPackageCallback(Callback):
    """Send the output of a forward job to a previous pipeline stage."""

    order = 5

    def after_compute(self):
        from pipegoose.nn.pipeline_parallel2._comm import send_package

        parallel_context = self.job.pipeline_context.parallel_context
        if parallel_context.pipeline_parallel_size > 1:
            output = self.job.output
            assert isinstance(output, Package), f"output must be an instance of Package, got {type(output)}"
            send_package(output, parallel_context)


class BackwardJob(Job):
    """Do backward pass."""

    def run_compute(self) -> torch.Tensor:
        microbatch_idx = self.input.metadata.microbatch_idx
        partition_idx = self.input.metadata.partition_idx
        prev_grad = self.input.data

        output = get_output_activations(microbatch_idx, partition_idx)
        input = get_input_activations(microbatch_idx, partition_idx)

        torch.autograd.backward(output, grad_tensors=prev_grad)

        if input.requires_grad is False:
            raise PipelineGradientFlowError("Gradients can't flow back to the input of the pipeline stage")

        # # TODO: remove this, since the grads is stored in module's weights
        # # and we do gradient accumulation, we don't need return grads or send to other stages
        # assert isinstance(input.grad, torch.Tensor)

        rank = self.pipeline_context.parallel_context.get_global_rank()
        print(f"executing backward job, rank={rank}, microbatch_idx={microbatch_idx}, partition_idx={partition_idx}")
        print(f"yay! gradients: {input.grad.shape}")

        if input.grad is None:
            raise PipelineGradientFlowError("Gradients can't flow back to the input of the pipeline stage")

        return input.grad
