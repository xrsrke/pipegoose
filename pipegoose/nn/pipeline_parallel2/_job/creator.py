from abc import ABC, abstractmethod

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.pipeline_parallel2._package import Package
from pipegoose.nn.pipeline_parallel2._job.job import Job, ForwardJob, BackwardJob
from pipegoose.nn.pipeline_parallel2._job.job_type import JobType
from pipegoose.nn.pipeline_parallel2._job.callback import Callback


class CreateForwardOutputPackage(Callback):
    """Create a new package for the output of a forward job."""

    def after_compute(self):
        data = self.job.raw_output
        orig_metadata = self.job.input.metadata

        package = Package(data, orig_metadata)
        package.metadata.partition_idx += 1

        self.job.output = package


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

    CBS = [CreateForwardOutputPackage, SendForwardPackage]

    @classmethod
    def create(cls, package: Package, parallel_context: ParallelContext) -> ForwardJob:
        assert isinstance(package, Package), f"package must be an instance of Package, got {type(package)}"

        job = ForwardJob(package)
        job.add_cbs(cls.CBS)

        return job


class _BackwardJobCreator(JobCreator):
    CBS = [CreateBackwardOutputPackage, SendBackwardPackage]

    @classmethod
    def create(cls, package: Package, parallel_context: ParallelContext) -> BackwardJob:
        assert isinstance(package, Package), f"package must be an instance of Package, got {type(package)}"

        job = BackwardJob(package)
        job.add_cbs(cls.CBS)

        return job


def create_job(package: Package, parallel_context: ParallelContext) -> Job:
    JOB_TYPE_TO_CREATOR = {
        JobType.FORWARD: _ForwardJobCreator,
        JobType.BACKWARD: _BackwardJobCreator,
    }

    job_type = package.metadata.job_type
    job = JOB_TYPE_TO_CREATOR[job_type].create(package, parallel_context)

    return job
