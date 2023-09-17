from abc import ABC, abstractmethod
from typing import Callable, Union

from pipegoose.nn.pipeline_parallel2._job.backward import (
    BackwardJob,
    CreateBackwardOutputPackageCallback,
    SendBackwardPackageCallback,
)
from pipegoose.nn.pipeline_parallel2._job.forward import (
    CreateForwardOutputPackageCallback,
    ForwardJob,
    SendForwardPackageCallback,
)
from pipegoose.nn.pipeline_parallel2._job.job import Job
from pipegoose.nn.pipeline_parallel2._job.job_type import JobType
from pipegoose.nn.pipeline_parallel2._package import Package
from pipegoose.nn.pipeline_parallel2.pipeline_context import PipelineContext


class JobCreator(ABC):
    """A base class for creating a job from a package."""

    @abstractmethod
    def create(self) -> Job:
        raise NotImplementedError("not implemented")


class _ForwardJobCreator(JobCreator):
    """Put a forward job into job queue for a worker to execute."""

    CBS = [CreateForwardOutputPackageCallback, SendForwardPackageCallback]

    @classmethod
    def create(cls, function: Callable, package: Package, pipeline_context: PipelineContext) -> ForwardJob:
        job = ForwardJob(function, package, cbs=cls.CBS, pipeline_context=pipeline_context)
        return job


class _BackwardJobCreator(JobCreator):
    CBS = [CreateBackwardOutputPackageCallback, SendBackwardPackageCallback]

    @classmethod
    def create(cls, function: Callable, package: Package, pipeline_context: PipelineContext) -> BackwardJob:
        job = BackwardJob(function, package, cbs=cls.CBS, pipeline_context=pipeline_context)
        return job


def create_job(function: Callable, package: Package, pipeline_context: PipelineContext) -> Union[ForwardJob, BackwardJob]:
    """Create a job based on the package."""
    assert isinstance(package, Package), f"package must be an instance of Package, got {type(package)}"
    assert isinstance(
        pipeline_context, PipelineContext
    ), f"pipeline_context must be an instance of PipelineContext, got {type(pipeline_context)}"

    JOB_TYPE_TO_CREATOR = {
        JobType.FORWARD: _ForwardJobCreator,
        JobType.BACKWARD: _BackwardJobCreator,
    }

    job_type = package.metadata.job_type
    job = JOB_TYPE_TO_CREATOR[job_type].create(function, package, pipeline_context)

    return job


def create_backwardable_forward_job():
    """
    Create a forward job that automatically schedules
    a backward job if you call forward(input).backward()
    """
