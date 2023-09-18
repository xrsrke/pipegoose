import torch

from pipegoose.nn.pipeline_parallel2._job.callback import Callback
from pipegoose.nn.pipeline_parallel2._job.job import Job
from pipegoose.nn.pipeline_parallel2._package import Package


class CreateBackwardOutputPackageCallback(Callback):
    """Create a new package for the output of a backward job."""

    def after_compute(self):
        data = self.job.output
        orig_metadata = self.job.input.metadata

        package = Package(data, orig_metadata)
        package.metadata.partition_idx -= 1

        self.job.output = package


class SendBackwardPackageCallback(Callback):
    pass


class BackwardJob(Job):
    """Do backward pass."""

    def run_compute(self) -> torch.Tensor:
        # key = self.job.key
        # activations = get_saved_activations(key)

        # grad_output = self.input.data

        # if activations.requires_grad:
        #     with torch.enable_grad():
        #         torch.autograd.backward(activations, grad_output)

        # return activations.grad
        return self.function(self.input.data)
