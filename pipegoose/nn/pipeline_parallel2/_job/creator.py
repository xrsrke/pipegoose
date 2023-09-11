from abc import ABC, abstractmethod

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.pipeline_parallel2._package import Package
from pipegoose.nn.pipeline_parallel2._job.job import Job, ForwardJob, BackwardJob
from pipegoose.nn.pipeline_parallel2._job.job_type import JobType


class JobCreator(ABC):
    """A base class for creating a job from a package."""

    @abstractmethod
    def create(self) -> Job:
        raise NotImplementedError("not implemented")


class _ForwardJobCreator(JobCreator):
    """Put a forward job into job queue for a worker to execute."""

    @staticmethod
    def create(package: Package, parallel_context: ParallelContext) -> ForwardJob:
        assert isinstance(package, Package), f"package must be an instance of Package, got {type(package)}"
        return ForwardJob(package)


class _BackwardJobCreator(JobCreator):
    @staticmethod
    def create(package: Package, parallel_context: ParallelContext) -> BackwardJob:
        assert isinstance(package, Package), f"package must be an instance of Package, got {type(package)}"
        return BackwardJob(package)


def create_job(package: Package, parallel_context: ParallelContext) -> Job:
    JOB_TYPE_TO_CREATOR = {
        JobType.FORWARD: _ForwardJobCreator,
        JobType.BACKWARD: _BackwardJobCreator,
    }

    job_type = package.metadata.job_type
    job = JOB_TYPE_TO_CREATOR[job_type].create(package, parallel_context)

    return job
