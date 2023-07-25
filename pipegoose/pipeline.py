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
        partitions: List[nn.Sequential],
        devices: Optional[List[torch.device]] = None,
        scheduler: BaseScheduler = DetermisticScheduler(),
    ) -> None:
        """Initialize the pipeline.

        Args:
            batches (List[Batch]): A list of micro-batches.
            partitions (List[nn.Sequential]): A partitioned model.
            devices (Optional[List[torch.device]], optional): A list of devices. Defaults to None.
            scheduler (BaseScheduler, optional): _description_. Defaults to DetermisticScheduler().
        """
        self.batches = batches
        self.partitions = partitions
        self.devices = devices
        self.scheduler = scheduler

    def fit(self):
        batches = self.batches
        partitions = self.partitions
        devices = self.devices
        scheduler = self.scheduler

        n_batches = len(batches)
        n_partitions = len(partitions)

        with spawn_worker(devices) as (in_queues, out_queues):
            for schedule in scheduler.generate(n_batches, n_partitions):
                self._depend(schedule)
                self._compute(schedule, in_queues, out_queues)

    def _depend(self, schedule: List[Tuple[int, int]]):
        """Enforce the dependency between batches and partitions."""
        batches = self.batches

        for microbatch_idx, partition_idx in schedule:
            if microbatch_idx != 0:
                create_backward_dependency(batches[microbatch_idx - 1], batches[microbatch_idx])

    def _compute(self, schedule: List[Tuple[int, int]], in_queues: List[Queue], out_queues: List[Queue]):
        """Compute the partitions."""
        batches = self.batches
        partitions = self.partitions

        for microbatch_idx, partition_idx in schedule:
            batch = batches[microbatch_idx]
            partrition = partitions[partition_idx]

            def compute(batch, partrition):
                def wrapper():
                    return partrition(batch)

                return wrapper

            task = Task(compute=compute(batch, partrition))
            in_queues[partition_idx].put(task)

        for microbatch_idx, partition_idx in schedule:
            queue_output = out_queues[partition_idx].get()
            task, output = queue_output.task, queue_output.output

            # put the output back to the batch
            batches[microbatch_idx] = output
