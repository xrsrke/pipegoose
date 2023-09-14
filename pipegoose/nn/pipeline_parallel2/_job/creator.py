from typing import Union
from abc import ABC, abstractmethod

from pipegoose.nn.pipeline_parallel2.pipeline_context import PipelineContext
from pipegoose.nn.pipeline_parallel2._package import Package
from pipegoose.nn.pipeline_parallel2._job.job import Job, ForwardJob, BackwardJob
from pipegoose.nn.pipeline_parallel2._job.job_type import JobType
from pipegoose.nn.pipeline_parallel2._job.callback import Callback


class SetForwardFunctionCallback(Callback):
    """Set the forward function of a forward job to the partition forward function."""

    def after_create(self):
        pipeline_context = self.job.pipeline_context
        self.job.function = pipeline_context.get_partition_forward()


class CreateForwardOutputPackage(Callback):
    """Create a new package for the output of a forward job."""

    def after_compute(self):
        data = self.job.raw_output
        orig_metadata = self.job.input.metadata

        package = Package(data, orig_metadata)
        package.metadata.partition_idx += 1

        self.job.output = package


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
        data = self.job.raw_output
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
    assert isinstance(pipeline_context, PipelineContext), f"pipeline_context must be an instance of PipelineContext, got {type(pipeline_context)}"

    JOB_TYPE_TO_CREATOR = {
        JobType.FORWARD: _ForwardJobCreator,
        JobType.BACKWARD: _BackwardJobCreator,
    }

    job_type = package.metadata.job_type
    job = JOB_TYPE_TO_CREATOR[job_type].create(package, pipeline_context)

    return job
