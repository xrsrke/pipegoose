from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union

from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel2._job.callback import Callback
from pipegoose.nn.pipeline_parallel2._job.job import BackwardJob, ForwardJob, Job
from pipegoose.nn.pipeline_parallel2._job.job_type import JobType
from pipegoose.nn.pipeline_parallel2._package import Package
from pipegoose.nn.pipeline_parallel2.pipeline_context import PipelineContext


class SetForwardFunctionCallback(Callback):
    """Set the forward function of a forward job to the partition forward function."""

    def after_create(self):
        pipeline_context = self.job.pipeline_context
        self.job.function = pipeline_context.get_partition_forward()


class CreateForwardOutputPackage(Callback):
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
        # so we hardcore and select that one
        # TODO: take into account that a pipeline stage can has more than one task
        # in a clock cycle, then find the correspond task to send the output to
        next_partition = next_schedule[0].partition_idx
        package.metadata.partition_idx = next_partition

        # NOTE: update the source and destination rank of the package

        return package

    def _update_src_and_dst_rank(self, package: Package) -> Package:
        pipeline_context = self.job.pipeline_context
        parallel_context = pipeline_context.parallel_context

        package.metadata.src = parallel_context.get_global_rank()
        package.metadata.dst = parallel_context.get_next_global_rank(ParallelMode.PIPELINE)

        return package


class SaveActivationIfTrainingCallback(Callback):
    """Save the activation of a forward job for backward pass if training."""

    order = 1

    def __init__(self):
        from pipegoose.nn.pipeline_parallel2.queue import ACTIVATIONS

        self.saved_activations = ACTIVATIONS

    def after_compute(self):
        if self.job.input.metadata.training.is_training is True:
            key = self.job.key
            self.saved_activations[key] = self.output


class SetBackwardFunctionCallback(Callback):
    def after_create(self):
        # TODO: change to autograd
        def wrapper(*args, **kwargs):
            import torch

            return torch.randn(1)

        self.job.function = wrapper


class CreateBackwardOutputPackage(Callback):
    """Create a new package for the output of a backward job."""

    def after_compute(self):
        data = self.job.output
        orig_metadata = self.job.input.metadata

        package = Package(data, orig_metadata)
        package.metadata.partition_idx -= 1

        self.job.output = package


class SendForwardPackage(Callback):
    def after_compute(self):
        pass


class SendBackwardPackage(Callback):
    pass


class JobCreator(ABC):
    """A base class for creating a job from a package."""

    @abstractmethod
    def create(self) -> Job:
        raise NotImplementedError("not implemented")


class _ForwardJobCreator(JobCreator):
    """Put a forward job into job queue for a worker to execute."""

    CBS = [SetForwardFunctionCallback, CreateForwardOutputPackage, SendForwardPackage]

    @classmethod
    def create(cls, package: Package, pipeline_context: PipelineContext) -> ForwardJob:
        assert isinstance(package, Package), f"package must be an instance of Package, got {type(package)}"

        job = ForwardJob(package, cbs=cls.CBS, pipeline_context=pipeline_context)

        return job


class _BackwardJobCreator(JobCreator):
    CBS = [SetBackwardFunctionCallback, CreateBackwardOutputPackage, SendBackwardPackage]

    @classmethod
    def create(cls, package: Package, pipeline_context: PipelineContext) -> BackwardJob:
        assert isinstance(package, Package), f"package must be an instance of Package, got {type(package)}"

        job = BackwardJob(package, cbs=cls.CBS, pipeline_context=pipeline_context)

        return job


def create_job(package: Package, pipeline_context: PipelineContext) -> Union[ForwardJob, BackwardJob]:
    assert isinstance(package, Package), f"package must be an instance of Package, got {type(package)}"
    assert isinstance(
        pipeline_context, PipelineContext
    ), f"pipeline_context must be an instance of PipelineContext, got {type(pipeline_context)}"

    JOB_TYPE_TO_CREATOR = {
        JobType.FORWARD: _ForwardJobCreator,
        JobType.BACKWARD: _BackwardJobCreator,
    }

    job_type = package.metadata.job_type
    job = JOB_TYPE_TO_CREATOR[job_type].create(package, pipeline_context)

    return job
