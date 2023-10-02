import torch

from pipegoose.nn.pipeline_parallel2._job.callback import Callback
from pipegoose.nn.pipeline_parallel2._job.job import Job
from pipegoose.nn.pipeline_parallel2._package import Package
from pipegoose.nn.pipeline_parallel2.queue import SavedActivation


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
        # print("doing backward job")
        # return self.function(self.input.data)

        microbatch_idx = self.input.metadata.microbatch_idx
        partition_idx = self.input.metadata.partition_idx
        key = SavedActivation.get_key(microbatch_idx, partition_idx)
        output = SavedActivation.get_saved_activations(key)
        prev_grad = self.input.data

        rank = self.pipeline_context.parallel_context.get_global_rank()

        print(f"executing backward job, rank={rank}, microbatch_idx={microbatch_idx}, partition_idx={partition_idx}")

        with torch.enable_grad():
            torch.autograd.backward(output, prev_grad)

        return output.grad
