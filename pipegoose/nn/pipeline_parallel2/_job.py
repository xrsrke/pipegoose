from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from threading import Thread

from pipegoose.nn.pipeline_parallel2.batch import Batch


class JobStatus(Enum):
    EXECUTING = auto()
    SUCCEED = auto()


@dataclass
class Metadata:
    # data
    microbatch_idx: int
    partition_idx: int

    # job
    is_forward: bool
    is_training: bool
    is_grad_enabled: bool  # not implemented yet
    is_fp16: bool  # not implemented yet
    status: JobStatus

    # communication
    src: int
    dst: int


class Job(ABC):
    def __init__(self, input: Batch, meta: Metadata) -> None:
        self._meta = meta
        self.input = input

    @abstractmethod
    def compute(self):
        raise NotImplementedError("not implemented")

    def meta(self):
        return self._meta


class ForwardJob(Job):
    pass


class BackwardJob(Job):
    pass


class JobRegister(ABC):
    def __init__(self, data):
        pass


class JobSelector(Thread):
    pass
