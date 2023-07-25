from queue import Queue
from typing import List, Optional, Tuple

import torch
from torch import nn

from pipegoose.dependency import create_backward_dependency
from pipegoose.microbatch import Batch
from pipegoose.scheduler import BaseScheduler, DetermisticScheduler
from pipegoose.worker import Task, spawn_worker


class Pipeline:
    """A base class for pipeline."""

    def __init__(
        self,
        batches: List[Batch],
        partritions: List[nn.Sequential],
        devices: Optional[List[torch.device]] = None,
        scheduler: BaseScheduler = DetermisticScheduler(),
    ) -> None:
        """Initialize the pipeline.

        Args:
            batches (List[Batch]): A list of micro-batches.
            partritions (List[nn.Sequential]): A partitioned model.
            devices (Optional[List[torch.device]], optional): A list of devices. Defaults to None.
            scheduler (BaseScheduler, optional): _description_. Defaults to DetermisticScheduler().
        """
        self.batches = batches
        self.partritions = partritions
        self.devices = devices
        self.scheduler = scheduler

    def fit(self):
        batches = self.batches
        partritions = self.partritions
        devices = self.devices
        scheduler = self.scheduler

        n_batches = len(batches)
        n_partritions = len(partritions)

        with spawn_worker(devices) as (in_queues, out_queues):
            for schedule in scheduler.generate(n_batches, n_partritions):
                self._depend(schedule)
                self._compute(schedule, in_queues, out_queues)

    def _depend(self, schedule: List[Tuple[int, int]]):
        """Enforce the dependency between batches and partritions."""
        batches = self.batches

        for microbatch_idx, partrition_idx in schedule:
            if microbatch_idx != 0:
                create_backward_dependency(batches[microbatch_idx - 1], batches[microbatch_idx])

    def _compute(self, schedule: List[Tuple[int, int]], in_queues: List[Queue], out_queues: List[Queue]):
        """Compute the partritions."""
        batches = self.batches
        partritions = self.partritions

        for microbatch_idx, partrition_idx in schedule:
            batch = batches[microbatch_idx]
            partrition = partritions[partrition_idx]

            def compute(batch, partrition):
                def wrapper():
                    return partrition(batch)

                return wrapper

            task = Task(compute=compute(batch, partrition))
            in_queues[partrition_idx].put(task)

        for microbatch_idx, partrition_idx in schedule:
            queue_output = out_queues[partrition_idx].get()
            task, output = queue_output.task, queue_output.output

            # put the output back to the batch
            batches[microbatch_idx] = output
