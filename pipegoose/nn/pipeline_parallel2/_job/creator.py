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


class ForwardJobCreator(JobCreator):
    """Put a forward job into job queue for a worker to execute."""

    @staticmethod
    def create(package: Package, parallel_context: ParallelContext) -> ForwardJob:
        assert isinstance(package, Package), f"package must be an instance of Package, got {type(package)}"
        return ForwardJob(package)


class BackwardJobCreator(JobCreator):
    @staticmethod
    def create(package: Package, parallel_context: ParallelContext) -> BackwardJob:
        assert isinstance(package, Package), f"package must be an instance of Package, got {type(package)}"
        return BackwardJob(package)


class JobCreator:
    """Create a job from a package that received from another pipeline stage."""

    JOB_FACTORY = {
        JobType.FORWARD: ForwardJobCreator,
        JobType.BACKWARD: BackwardJobCreator,
    }

    def __init__(self, parallel_context: ParallelContext):
        self.parallel_context = parallel_context

    def create(self, package: Package) -> Job:
        assert isinstance(package, Package), f"package must be an instance of Package, got {type(package)}"

        job_type = package.metadata.job_type
        job = self.JOB_FACTORY[job_type].create(package, self.parallel_context)

        return job
