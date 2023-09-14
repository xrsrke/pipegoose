from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.pipeline_parallel2._job.job_type import JobType
from pipegoose.nn.pipeline_parallel2._worker import BaseWorkerManager
from pipegoose.nn.pipeline_parallel2.pipeline_context import PipelineContext
from pipegoose.nn.pipeline_parallel2.scheduler import BaseScheduler
from pipegoose.nn.pipeline_parallel.partrition import BasePartitioner


@dataclass
class Schedule:
    job_type: JobType
    partition_idx: int
    microbatch_idx: int


class PipelineEngine:
    """A base pipeline engine that can be used to implement different pipeline engines."""

    def __init__(
        self,
        module: nn.Module,
        partitioner: BasePartitioner,
        scheduler: BaseScheduler,
        worker_manager: BaseWorkerManager,
        parallel_context: ParallelContext,
    ):
        assert isinstance(module, nn.Module), f"module must be an instance of nn.Module, got {type(module)}"
        assert isinstance(
            parallel_context, ParallelContext
        ), f"parallel_context must be an instance of ParallelContext, got {type(parallel_context)}"

        self.module = module
        self.partitioner = partitioner
        self.scheduler = scheduler
        self.worker_manager = worker_manager
        self.parallel_context = parallel_context

        self.pipeline_context = PipelineContext(self.scheduler, self.parallel_context)

    def forward(self, input: torch.Tensor):
        with self.worker_manager.spawn(self.pipeline_context):
            for schedule in self.scheduler.generate():
                self._compute(schedule)

    def _compute(self, batches: torch.Tensor, schedule: List[Schedule]):
        def get_current_node_schedule():
            pass

        schedule = get_current_node_schedule(schedule)

    def _construct_first_job(self):
        pass
