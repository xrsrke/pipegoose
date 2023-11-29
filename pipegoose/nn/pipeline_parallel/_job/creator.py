from abc import ABC, abstractmethod
from typing import Callable, Union

import torch

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.pipeline_parallel._job.backward import (
    BackwardJob,
    CreateBackwardOutputPackageCallback,
    SendBackwardPackageCallback,
    save_grad_loss,
)
from pipegoose.nn.pipeline_parallel._job.callback import Callback
from pipegoose.nn.pipeline_parallel._job.forward import (
    ConfirmCompleteATaskToProgressTracker,
    CreateForwardOutputPackageCallback,
    ForwardJob,
    SaveBufferForBackwardCallback,
    SendForwardPackageCallback,
)
from pipegoose.nn.pipeline_parallel._job.job import Job
from pipegoose.nn.pipeline_parallel._job.job_type import JobType
from pipegoose.nn.pipeline_parallel._package import Metadata, Package
from pipegoose.nn.pipeline_parallel.pipeline_context import PipelineContext
from pipegoose.nn.pipeline_parallel.queue import JobQueue


class JobCreator(ABC):
    """A base class for creating a job from a package."""

    @abstractmethod
    def create(self) -> Job:
        raise NotImplementedError("not implemented")


class ScheduleBackwardJobCallback(Callback):
    order = 3

    def __init__(self, pipeline_context: PipelineContext):
        self.pipeline_context = pipeline_context

    def after_compute(self):
        pass

        package = self.job.output
        microbatch_idx = self.job.input.metadata.microbatch_idx
        partition_idx = self.job.input.metadata.partition_idx

        # NOTE: only the last stage needs to save the gradient loss
        if self.pipeline_context.is_last_stage:
            if package.metadata.microbatch_idx == self.pipeline_context.num_microbatches - 1:
                new_package = schedule_backward_execution(package, self.pipeline_context)
                self.job.output = new_package
            else:
                new_package = save_grad_loss(package)
                self.job.output = new_package
        else:
            new_package = self.job.output

        from pipegoose.nn.pipeline_parallel.queue import _SAVED_SCHEDULED_ACTIVATIONS

        _SAVED_SCHEDULED_ACTIVATIONS[(microbatch_idx, partition_idx)] = new_package.data


class _ForwardJobCreator(JobCreator):
    """Create a forward job for pipeline parallelism."""

    @classmethod
    def create(
        cls, function: Callable, package: Package, parallel_context: ParallelContext, pipeline_context: PipelineContext
    ) -> ForwardJob:
        callbacks = [
            CreateForwardOutputPackageCallback(parallel_context, pipeline_context),
            SaveBufferForBackwardCallback(),
            ScheduleBackwardJobCallback(pipeline_context),
            # SaveActivationIfTrainingCallback(),
            # SaveInputActivationsCallback(),
            SendForwardPackageCallback(parallel_context),
            ConfirmCompleteATaskToProgressTracker(parallel_context),
        ]
        job = ForwardJob(function, package, callbacks)
        return job


class _BackwardJobCreator(JobCreator):
    """Create a backward job for pipeline parallelism."""

    @classmethod
    def create(
        cls, function: Callable, package: Package, parallel_context: ParallelContext, pipeline_context: PipelineContext
    ) -> BackwardJob:
        from pipegoose.nn.pipeline_parallel.queue import (
            InputActivations,
            SavedActivation,
        )

        microbatch_idx = package.metadata.microbatch_idx
        partition_idx = package.metadata.partition_idx

        assert (
            SavedActivation.is_saved(microbatch_idx, partition_idx) is True
        ), f"No saved activations for \
            microbatch_idx={microbatch_idx}, partition_idx={partition_idx}"
        assert (
            InputActivations.is_saved(microbatch_idx, partition_idx) is True
        ), f"No saved input activations for \
            microbatch_idx={microbatch_idx}, partition_idx={partition_idx}"

        callbacks = [
            CreateBackwardOutputPackageCallback(parallel_context, pipeline_context),
            SendBackwardPackageCallback(parallel_context),
            ConfirmCompleteATaskToProgressTracker(parallel_context),
        ]
        job = BackwardJob(function, package, is_scheduled=True, cbs=callbacks)
        return job


def create_job(
    function: Callable, package: Package, parallel_context: ParallelContext, pipeline_context: PipelineContext
) -> Union[ForwardJob, BackwardJob]:
    """Create a job based on the package."""
    assert isinstance(package, Package), f"package must be an instance of Package, got {type(package)}"
    assert isinstance(
        pipeline_context, PipelineContext
    ), f"pipeline_context must be an instance of PipelineContext, got {type(pipeline_context)}"

    JOB_TYPE_TO_CREATOR = {
        JobType.FORWARD: _ForwardJobCreator,
        JobType.BACKWARD: _BackwardJobCreator,
    }

    job_type = package.metadata.job_type
    job = JOB_TYPE_TO_CREATOR[job_type].create(function, package, parallel_context, pipeline_context)

    return job


def schedule_backward_execution(package: Package, pipeline_context: PipelineContext) -> Package:
    class BackwardCoordinationFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, metadata: Metadata, input: torch.Tensor) -> torch.Tensor:
            ctx.package_meta = metadata
            return input

        @staticmethod
        def backward(ctx, grad_input: torch.Tensor) -> (None, torch.Tensor):
            metadata = ctx.package_meta
            print("trigger creating backward job")
            _run_backward_execution(grad_input, metadata)
            return (None, None)

    data = package.data
    new_data = BackwardCoordinationFunction.apply(package.metadata, data)
    package.data = new_data
    return package


def _run_backward_execution(grad_input, metadata):
    import torch.distributed as dist

    def backward_function(self):
        pass

    dist.barrier()

    pipeline_context = PipelineContext.get_context()
    parallel_context = ParallelContext.get_context()
    pipeline_context.backward()

    from pipegoose.nn.pipeline_parallel.sync.handshake import get_progress_tracker
    from pipegoose.nn.pipeline_parallel.sync.progress_tracker import (
        get_progresses_from_pipeline_context,
    )

    progress = get_progresses_from_pipeline_context(pipeline_context)
    progress_tracker = get_progress_tracker()

    if parallel_context.get_global_rank() == 0:
        progress = get_progresses_from_pipeline_context(pipeline_context)
        progress_tracker.initiate(progress)

    rank = parallel_context.get_global_rank()

    dist.barrier()

    for tasks in pipeline_context.get_schedule():
        dist.barrier()

        print(f"rank={rank}, entered clock_idx: {pipeline_context.clock_idx}")

        if len(tasks) > 0:
            for task in tasks:
                microbatch_idx = task.microbatch_idx
                partition_idx = task.partition_idx

                print(
                    f"rank={rank}, clock_idx={pipeline_context.clock_idx}, microbatch_idx={microbatch_idx}, partition_idx={partition_idx}"
                )

                if pipeline_context.is_last_stage:
                    if pipeline_context.is_last_microbatch(microbatch_idx) is False:
                        from pipegoose.nn.pipeline_parallel.queue import (
                            _SAVED_GRAD_LOSS,
                            _SAVED_METADATA_of_GRAD_LOSS,
                        )

                        grad_input = _SAVED_GRAD_LOSS[(microbatch_idx, partition_idx)]
                        metadata = _SAVED_METADATA_of_GRAD_LOSS[(microbatch_idx, partition_idx)]

                    package = Package(grad_input, metadata)
                    package.metadata.job_type = JobType.BACKWARD
                else:
                    from pipegoose.nn.pipeline_parallel._comm import RECV_QUEUE

                    package = RECV_QUEUE.get()

                backward_job = create_job(backward_function, package, parallel_context, pipeline_context)
                # NOTE : put the backward job to pending queue
                JobQueue.PENDING_JOBS.put(backward_job)

                microbatch_idx = metadata.microbatch_idx
                print(f"rank={rank}, created backward job: microbatch_idx={microbatch_idx}, partition_idx={partition_idx}")

        dist.barrier()

    dist.barrier()
