from typing import List

import torch
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.pipeline_parallel._utils import get_partition_idx
from pipegoose.nn.pipeline_parallel._worker import WorkerManager
from pipegoose.nn.pipeline_parallel.pipeline_engine import PipelineEngine
from pipegoose.nn.pipeline_parallel.scheduler import GPipeScheduler


class PipelineParallel:
    """Automatically parallelize a module using pipeline parallelism."""

    def __init__(
        self,
        modules: List[nn.Module],
        num_microbatches: int,
        parallel_context: ParallelContext,
    ):
        self.modules = modules
        self.num_microbatches = num_microbatches
        self.parallel_context = parallel_context

    @torch.no_grad()
    def parallelize(self) -> nn.Module:
        partition_idx = get_partition_idx(self.parallel_context)
        module = self.modules[partition_idx]

        n_partitions = self.parallel_context.pipeline_parallel_size
        scheduler = GPipeScheduler(self.num_microbatches, n_partitions)
        worker_manager = WorkerManager()

        pipeline_engine = PipelineEngine(
            module=module,
            scheduler=scheduler,
            worker_manager=worker_manager,
            parallel_context=self.parallel_context,
        )

        module.forward = pipeline_engine.run
        return module
