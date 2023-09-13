from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Optional

import torch

from pipegoose.nn.pipeline_parallel2._package import Package


class JobStatus(Enum):
    # NOTE: wait for a worker pick up this job and execute it
    PENDING = auto()  # just created and putted into job queue
    EXECUTING = auto()
    EXECUTED = auto()  # executed but not sent the output to another pipeline stage
    DONE = auto()  # executed and sent the output to another pipeline stage
    FAILED = auto()  # failed to execute


class Job(ABC):
    """A job that will be executed by a worker."""

    def __init__(self, package: Package):
        self.package = package
        self._status = JobStatus.PENDING
        self._output = None

        def generate_random_string(length=15):
            import random
            import string
            characters = string.ascii_letters + string.digits
            return ''.join(random.choice(characters) for i in range(length))

        self._key = generate_random_string()

    @property
    def status(self) -> JobStatus:
        return self._status

    @property
    def key(self) -> str:
        return self._key

    @property
    def output(self) -> Optional[Package]:
        return self._output

    def compute(self) -> Package:
        output = self.run_compute()

        self._status = JobStatus.EXECUTED
        package = self.construct_package(output)
        self._output = package

        return package

    @abstractmethod
    def run_compute(self):
        """The actual computation of this job."""
        raise NotImplementedError("not implemented")

    @abstractmethod
    def construct_package(self):
        """
        Construct a new package based on the output of a job,
        then send this package to another pipeline stage. The other pipeline stage
        will construct a job based on the metadata of the package.
        """
        raise NotImplementedError("not implemented")

    @abstractmethod
    def finalize(self):
        """Execute this method after `compute`."""
        raise NotImplementedError("not implemented")


class ForwardJob(Job):
    def run_compute(self) -> torch.Tensor:
        data = self.package.data
        return data

    def construct_package(self, data: torch.Tensor) -> Package:
        package = Package(data, self.package.metadata)
        package.metadata.partition_idx += 1
        return package

    def finalize(self):
        pass


class BackwardJob(Job):
    def run_compute(self) -> torch.Tensor:
        return self.package.data

    def construct_package(self, data: torch.Tensor) -> Package:
        package = Package(data, self.package.metadata)
        package.metadata.partition_idx -= 1
        return package

    def finalize(self):
        pass
