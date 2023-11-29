import torch
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.parallel import Parallel
from pipegoose.nn.pipeline_parallel._utils import get_partition_idx
from pipegoose.nn.pipeline_parallel._worker import WorkerManager
from pipegoose.nn.pipeline_parallel.partitioner import UniformPartitioner
from pipegoose.nn.pipeline_parallel.pipeline_engine import PipelineEngine
from pipegoose.nn.pipeline_parallel.scheduler import GPipeScheduler


class PipelineParallel(Parallel):
    """Automatically parallelize a module using pipeline parallelism."""

    def __init__(
        self,
        module: nn.Module,
        num_microbatches: int,
        parallel_context: ParallelContext,
    ):
        self.module = module
        self.num_microbatches = num_microbatches
        self.parallel_context = parallel_context

    @torch.no_grad()
    def parallelize(self) -> nn.Module:
        if self.parallel_context.pipeline_parallel_size > 1:
            partition_idx = get_partition_idx(self.parallel_context)
            n_partitions = self.parallel_context.pipeline_parallel_size
            partitions = UniformPartitioner(self.module, n_partitions=n_partitions).split(["input_ids"])
            module = partitions[partition_idx]

            scheduler = GPipeScheduler(self.num_microbatches, n_partitions)
            worker_manager = WorkerManager()

            pipeline_engine = PipelineEngine(
                module=module,
                scheduler=scheduler,
                worker_manager=worker_manager,
                parallel_context=self.parallel_context,
            )

            module.forward = pipeline_engine.run

            self._save_metadata(module, self.parallel_context)

            return module
        else:
            return self.module
