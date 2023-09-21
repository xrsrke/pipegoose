from dataclasses import dataclass

import torch
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode

# from pipegoose.nn.pipeline_parallel.partrition import BasePartitioner
from pipegoose.nn.pipeline_parallel2._job.creator import create_job
from pipegoose.nn.pipeline_parallel2._job.job_type import JobType
from pipegoose.nn.pipeline_parallel2._package import Metadata, Package, TrainingMetadata
from pipegoose.nn.pipeline_parallel2._worker import BaseWorkerManager
from pipegoose.nn.pipeline_parallel2.pipeline_context import PipelineContext
from pipegoose.nn.pipeline_parallel2.queue import JobQueue
from pipegoose.nn.pipeline_parallel2.scheduler import BaseScheduler


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
        # partitioner: BasePartitioner,
        scheduler: BaseScheduler,
        worker_manager: BaseWorkerManager,
        parallel_context: ParallelContext,
    ):
        assert isinstance(module, nn.Module), f"module must be an instance of nn.Module, got {type(module)}"
        assert isinstance(
            parallel_context, ParallelContext
        ), f"parallel_context must be an instance of ParallelContext, got {type(parallel_context)}"

        self.module = module
        # self.partitioner = partitioner
        self.scheduler = scheduler
        self.worker_manager = worker_manager
        self.parallel_context = parallel_context

        self.pipeline_context = PipelineContext(self.scheduler, self.parallel_context)

    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        self.worker_manager.spawn()
        n_microbatches = self.scheduler.n_microbatches

        # microbatches = microbatch.split(inputs, n_microbatches=self.scheduler.n_microbatches)
        microbatches = torch.chunk(inputs, chunks=n_microbatches, dim=0)

        if self.parallel_context.is_first_rank(ParallelMode.PIPELINE):
            for task in self.pipeline_context.schedule:
                if task.partition_idx == 0:
                    microbatch_idx = task.microbatch_idx

                    batch = microbatches[microbatch_idx]
                    forward_job = self._construct_first_job(microbatch_idx=microbatch_idx, input=batch)

                    JobQueue.PENDING_JOBS.put(forward_job)

    def _construct_first_job(self, microbatch_idx: int, input: torch.Tensor):
        PARTITION_IDX = 0
        IS_TRAINING = torch.is_grad_enabled()

        metadata = Metadata(
            microbatch_idx=microbatch_idx,
            partition_idx=PARTITION_IDX,
            job_type=JobType.FORWARD,
            training=TrainingMetadata(
                is_training=IS_TRAINING,
                is_grad_enabled=IS_TRAINING,
            ),
            src=self.parallel_context.get_global_rank(),
            dst=self.parallel_context.get_global_rank(),
        )
        package = Package(
            data=input,
            metadata=metadata,
        )

        function = nn.Linear(5, 5)
        job = create_job(function, package, self.pipeline_context)
        return job
