import torch
from torch import nn

from pipegoose.constants import PIPELINE_MAX_WORKERS, PIPELINE_MIN_WORKERS
from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.pipeline_parallel2._worker import WorkerManager
from pipegoose.nn.pipeline_parallel2.partitioner import PartitionPolicy
from pipegoose.nn.pipeline_parallel2.pipeline_engine import PipelineEngine
from pipegoose.nn.pipeline_parallel2.scheduler import SchedulerType, get_scheduler


class PipelineParallel:
    """Automatically parallelize a module using pipeline parallelism."""

    def __init__(
        self,
        module: nn.Module,
        num_microbatches: int,
        scheduler_type: SchedulerType,
        partition_policy: PartitionPolicy,
        parallel_context: ParallelContext,
    ):
        self.module = module
        self.num_microbatches = num_microbatches
        self.scheduler_type = scheduler_type
        self.partition_policy = partition_policy
        self.parallel_context = parallel_context

    @torch.no_grad()
    def parallelize(self) -> nn.Module:
        module = self.module

        # TODO: lazy init
        scheduler = get_scheduler(
            scheduler_type=self.scheduler_type,
            num_microbatches=self.num_microbatches,
            parallel_context=self.parallel_context,
        )
        worker_manager = WorkerManager(
            min_workers=PIPELINE_MIN_WORKERS,
            max_workers=PIPELINE_MAX_WORKERS,
            parallel_context=self.parallel_context,
        )
        pipeline_engine = PipelineEngine(
            module=module,
            scheduler=scheduler,
            worker_manager=worker_manager,
            parallel_context=self.parallel_context,
        )

        pipeline_engine.parallelize(module)

        return pipeline_engine
