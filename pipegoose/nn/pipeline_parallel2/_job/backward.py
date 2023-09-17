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
    def run_compute(self) -> torch.Tensor:
        return self.function(self.input.data)
