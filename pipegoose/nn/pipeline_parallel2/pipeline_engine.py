import time
from dataclasses import dataclass

import torch
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel2._comm import RECV_QUEUE
from pipegoose.nn.pipeline_parallel2._job.creator import create_job
from pipegoose.nn.pipeline_parallel2._job.job_type import JobType
from pipegoose.nn.pipeline_parallel2._package import Metadata, Package, TrainingMetadata
from pipegoose.nn.pipeline_parallel2._worker import BaseWorkerManager
from pipegoose.nn.pipeline_parallel2.pipeline_context import PipelineContext
from pipegoose.nn.pipeline_parallel2.queue import JobQueue
from pipegoose.nn.pipeline_parallel2.scheduler import BaseScheduler
from pipegoose.nn.pipeline_parallel2.sync.callback import Callback
from pipegoose.nn.pipeline_parallel2.sync.handshake import (
    ProgressTracker,
    set_progress_tracker,
)


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
        rank: int,
        partition_func,
    ):
        assert isinstance(module, nn.Module), f"module must be an instance of nn.Module, got {type(module)}"
        # assert isinstance(
        #     parallel_context, ParallelContext
        # ), f"parallel_context must be an instance of ParallelContext, got {type(parallel_context)}"

        self.module = module
        # self.partitioner = partitioner
        self.scheduler = scheduler
        self.worker_manager = worker_manager
        self.parallel_context = parallel_context

        self.pipeline_context = PipelineContext(scheduler, parallel_context)
        self.rank = rank
        self.partition_func = partition_func

    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        MASTER_RANK = 0

        # from hanging_threads import start_monitoring
        # monitoring_thread = start_monitoring()

        self.worker_manager.spawn()
        n_microbatches = self.scheduler.n_microbatches

        # microbatches = microbatch.split(inputs, n_microbatches=n_microbatches)
        microbatches = torch.chunk(inputs, chunks=n_microbatches, dim=0)

        # NOTE: add a callback to the progress tracker
        # that if the clock_idx is increased, then
        # notify pipeline_context to yield the next schedule
        class IncreasePipelineContextClockCycleCallback(Callback):
            def __init__(self, pipeline_context):
                self.pipeline_context = pipeline_context

            def after_new_clock_cycle(self, progress, clock_idx):
                parallel_context = self.pipeline_context.parallel_context
                print(f"increase clock, clock_idx={clock_idx}, rank={parallel_context.get_local_rank(ParallelMode.GLOBAL)}")
                self.pipeline_context.increase_a_clock_cycle()
                time.sleep(1)

        callbacks = [IncreasePipelineContextClockCycleCallback(self.pipeline_context)]
        progress_tracker = ProgressTracker(
            MASTER_RANK, callbacks=callbacks, parallel_context=self.parallel_context, parallel_mode=ParallelMode.GLOBAL
        )
        # NOTE: wait for all ranks to be initiated
        time.sleep(1)

        if self.parallel_context.is_first_rank(ParallelMode.PIPELINE):
            schedules = self.pipeline_context.schedules
            progress = {
                i: {(item.microbatch_idx, item.partition_idx): False for item in sublist}
                for i, sublist in enumerate(schedules)
            }
            progress_tracker.initiate(progress)

        time.sleep(1)

        set_progress_tracker(progress_tracker)

        time.sleep(1)

        # from hanging_threads import start_monitoring
        # monitoring_thread = start_monitoring()

        for tasks in self.pipeline_context.get_schedule():

            time.sleep(2)
            rank = self.parallel_context.get_local_rank(ParallelMode.GLOBAL)
            partition_idx = self.pipeline_context.partition_idx

            if rank == 0:
                assert 1 == 1

            if self.pipeline_context.clock_idx == 1:
                assert 1 == 1

            if len(tasks) > 0:
                print(f"[enter look] clock_idx={self.pipeline_context.clock_idx}, rank={rank}, partition_idx={partition_idx}")
                for task in tasks:
                    microbatch_idx = task.microbatch_idx
                    partition_idx = task.partition_idx
                    if self.parallel_context.is_first_rank(ParallelMode.PIPELINE):
                        if partition_idx == 0:
                            batch = microbatches[microbatch_idx]
                            package = self._construct_first_package(microbatch_idx, input=batch)
                    else:
                        package = RECV_QUEUE.get()

                    print(
                        f"[received a package]clock_idx={self.pipeline_context.clock_idx}, rank={rank}, partition_idx={partition_idx}",
                        package.metadata,
                    )

                    job = create_job(self.partition_func, package, self.pipeline_context)

                    # print(f"created a job: {package.metadata}")

                    JobQueue.PENDING_JOBS.put(job)
            time.sleep(2)

    # def _retrieve_package_from_received_package(self, microbatch_idx, partition_idx):
    #     # package = RECV_QUEUE[(microbatch_idx, partition_idx)]
    #     package = RECV_QUEUE.get()
    #     return package

    def _construct_first_package(self, microbatch_idx: int, input: torch.Tensor):
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
