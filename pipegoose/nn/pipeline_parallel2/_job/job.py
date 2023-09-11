from abc import ABC, abstractmethod
from enum import Enum, auto

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
    def __init__(self, package: Package) -> None:
        self.package = package
        self._status = JobStatus.PENDING

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

    def compute(self) -> torch.Tensor:
        output = self.run_compute()
        self._status = JobStatus.EXECUTED
        return output

    @abstractmethod
    def run_compute(self):
        """The actual computation of this job."""
        raise NotImplementedError("not implemented")

    @abstractmethod
    def finalize(self):
        """Execute this method after `compute`."""
        raise NotImplementedError("not implemented")


class ForwardJob(Job):
    def run_compute(self) -> torch.Tensor:
        return self.package.data

    def finalize(self):
        pass


class BackwardJob(Job):
    def run_compute(self) -> torch.Tensor:
        return self.package.data

    def finalize(self):
        pass
