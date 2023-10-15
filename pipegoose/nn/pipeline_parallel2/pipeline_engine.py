from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel2._comm import RECV_QUEUE
from pipegoose.nn.pipeline_parallel2._job.creator import create_job
from pipegoose.nn.pipeline_parallel2._job.job_type import JobType
from pipegoose.nn.pipeline_parallel2._package import Metadata, Package, TrainingMetadata
from pipegoose.nn.pipeline_parallel2._worker import BaseWorkerManager
from pipegoose.nn.pipeline_parallel2.pipeline_context import PipelineContext
from pipegoose.nn.pipeline_parallel2.queue import (
    JobQueue,
    get_output_activations,
    save_input_activations,
)
from pipegoose.nn.pipeline_parallel2.scheduler import BaseScheduler
from pipegoose.nn.pipeline_parallel2.sync.callback import Callback
from pipegoose.nn.pipeline_parallel2.sync.handshake import (
    ProgressTracker,
    set_progress_tracker,
)
from pipegoose.nn.pipeline_parallel2.sync.progress_tracker import (
    get_progresses_from_pipeline_context,
)


@dataclass
class Schedule:
    job_type: JobType
    partition_idx: int
    microbatch_idx: int


class PipelineEngine:
    """A base pipeline engine that can be used to implement different pipeline engines."""

    MASTER_RANK = 0

    def __init__(
        self,
        module: nn.Module,
        # partitioner: BasePartitioner,
        scheduler: BaseScheduler,
        worker_manager: BaseWorkerManager,
        parallel_context: ParallelContext,
        partition_func,
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

        self.pipeline_context = PipelineContext(scheduler, parallel_context)
        self.partition_func = partition_func

    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        self.worker_manager.spawn()
        n_microbatches = self.scheduler.n_microbatches

        # microbatches = microbatch.split(inputs, n_microbatches=n_microbatches)
        microbatches = torch.chunk(inputs, chunks=n_microbatches, dim=0)

        # NOTE: add a callback to the progress tracker
        # that if the clock_idx is increased, then
        # notify pipeline_context to yield the next schedule
        class IncreasePipelineContextClockCycleCallback(Callback):
            def __init__(self, parallel_context, pipeline_context):
                self.parallel_context = parallel_context
                self.pipeline_context = pipeline_context

            def after_new_clock_cycle(self, progress, clock_idx):
                # NOTE: suppose we have tensor_parallel_size = 3
                # that means a pipeline stage is split into 3 slices
                # we want only one slice to increase the clock
                # here we choose the last slice to increase the clock
                if self.parallel_context.is_last_rank(ParallelMode.TENSOR):
                    print(
                        f"increase clock, clock_idx={clock_idx}, rank={self.parallel_context.get_local_rank(ParallelMode.GLOBAL)}"
                    )
                    self.pipeline_context.increase_a_clock_cycle()

        callbacks = [IncreasePipelineContextClockCycleCallback(self.parallel_context, self.pipeline_context)]
        progress_tracker = ProgressTracker(
            self.MASTER_RANK, callbacks=callbacks, parallel_context=self.parallel_context, parallel_mode=ParallelMode.GLOBAL
        )
        # # NOTE: wait for all ranks to be initiated
        dist.barrier()

        if self.parallel_context.get_global_rank() == 0:
            progress = get_progresses_from_pipeline_context(self.pipeline_context)
            progress_tracker.initiate(progress)
            print(progress)

        dist.barrier()

        set_progress_tracker(progress_tracker)
        self.pipeline_context.forward()

        for tasks in self.pipeline_context.get_schedule():
            dist.barrier()

            if len(tasks) > 0:
                for task in tasks:
                    microbatch_idx = task.microbatch_idx
                    partition_idx = task.partition_idx
                    if self.parallel_context.is_first_rank(ParallelMode.PIPELINE):
                        if partition_idx == 0:
                            batch = microbatches[microbatch_idx]
                            package = self._construct_first_package(microbatch_idx, input=batch)
                    else:
                        package = RECV_QUEUE.get()

                    save_input_activations(package.data, microbatch_idx=microbatch_idx, partition_idx=partition_idx)

                    job = create_job(self.partition_func, package, self.parallel_context, self.pipeline_context)
                    JobQueue.PENDING_JOBS.put(job)

            dist.barrier()

        dist.barrier()

        if self.pipeline_context.is_last_stage:
            outputs = []
            for microbatch_idx in range(n_microbatches):
                outputs.append(get_output_activations(microbatch_idx, self.pipeline_context.partition_idx))

            outputs = torch.cat(outputs, dim=0)
            return outputs

    def _construct_first_package(self, microbatch_idx: int, input: torch.Tensor) -> Package:
        """Construct the first forward package of a microbatch."""
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
            dst=self.parallel_context.get_next_global_rank(ParallelMode.PIPELINE),
        )
        package = Package(
            data=input,
            metadata=metadata,
        )

        return package
