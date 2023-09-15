from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Optional

import torch

from pipegoose.nn.pipeline_parallel2._job.callback import Callback, CallbackEvent
from pipegoose.nn.pipeline_parallel2._package import Package
from pipegoose.nn.pipeline_parallel2.pipeline_context import PipelineContext


class JobStatus(Enum):
    # NOTE: wait for a worker pick up this job and execute it
    PENDING = auto()  # just created and putted into job queue
    EXECUTING = auto()
    EXECUTED = auto()  # executed but not sent the output to another pipeline stage
    DONE = auto()  # executed and sent the output to another pipeline stage
    FAILED = auto()  # failed to execute


class Job(ABC):
    """A job that will be executed by a worker."""

    def __init__(self, input: Package, cbs: List[Callback] = [], pipeline_context: PipelineContext = None):
        assert isinstance(
            pipeline_context, PipelineContext
        ), f"input must be an instance of PipelineContext, got {type(input)}"

        self.input = input
        self.cbs = []
        self.pipeline_context = pipeline_context

        self._status = JobStatus.PENDING
        self._output = None

        def generate_random_string(length=15):
            import random
            import string

            characters = string.ascii_letters + string.digits
            return "".join(random.choice(characters) for i in range(length))

        self._key = generate_random_string()

        self.add_cbs(cbs)
        self._run_callback(CallbackEvent.AFTER_CREATE)

    @property
    def status(self) -> JobStatus:
        return self._status

    @property
    def key(self) -> str:
        return self._key

    @property
    def output(self) -> Optional[Package]:
        return self._output

    @output.setter
    def output(self, value: Optional[Package]):
        self._output = value

    def compute(self) -> Optional[Package]:
        try:
            self._run_callback(CallbackEvent.BEFORE_COMPUTE)

            # TODO: refactor make other callbacks to be able to access the output of a job
            self._output = self.run_compute()

            # TODO: turn the update of job status into a callback
            self._status = JobStatus.EXECUTED

            self._run_callback(CallbackEvent.AFTER_COMPUTE)

            return self.output
        except Exception as e:
            raise e

    def add_cbs(self, cbs: List[Callback]):
        """Add a list of callbacks to this job."""
        for cb in cbs:
            self.add_cb(cb)

    def remove_cbs(self, cbs: List[Callback]):
        for cb in cbs:
            self.remove_cb(cb)

    def add_cb(self, cb: Callback):
        """Add a callback to this job."""
        if isinstance(cb, type):
            cb = cb()

        assert isinstance(cb, Callback), f"cb must be an instance of Callback, got {type(cb)}"

        # NOTE: lets the callback access the job attributes
        cb.job = self
        self.cbs.append(cb)

    def remove_cb(self, cb: Callback):
        """Remove a callback from this job."""
        # NOTE: if cb is a class
        if isinstance(cb, type):
            cbs = [x for x in self.cbs if isinstance(x, cb)]
            self.remove_cbs(cbs)
        else:
            if cb in self.cbs:
                self.cbs.remove(cb)

    def _run_callback(self, event_name: str):
        sorted_cbs = sorted(self.cbs, key=lambda x: x.order)
        # NOTE: get the value of an enum member
        event_name = event_name.value

        for cb in sorted_cbs:
            event_method = getattr(cb, event_name, None)
            if event_method is not None:
                event_method()

    @abstractmethod
    def run_compute(self):
        """The actual computation of this job."""
        raise NotImplementedError("not implemented")


class ForwardJob(Job):
    def run_compute(self) -> torch.Tensor:
        return self.input.data


class BackwardJob(Job):
    def run_compute(self) -> torch.Tensor:
        return self.input.data
