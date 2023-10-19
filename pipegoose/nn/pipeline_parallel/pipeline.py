from torch import nn

from pipegoose.constants import PIPELINE_MAX_WORKERS, PIPELINE_MIN_WORKERS
from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.pipeline_parallel.scheduler import SchedulerType, get_scheduler


class _PipelineEngine:
    """Turn a 🤗 transformers model into a pipeline parallel model."""

    def __init__(
        self,
        module: nn.Module,
        num_concurrent: int = PIPELINE_MIN_WORKERS,
        max_concurrent: int = PIPELINE_MAX_WORKERS,
        scheduler: SchedulerType = SchedulerType.GPIPE,
        parallel_context: ParallelContext = None,
    ):
        assert num_concurrent <= max_concurrent, "num_concurrent must be less than or equal to max_concurrent"
        assert parallel_context is not None, "parallel_context must be provided"

        assert isinstance(
            parallel_context, ParallelContext
        ), f"parallel_context must be an instance of ParallelContext, got {type(parallel_context)}"
        assert isinstance(module, nn.Module), f"module must be an instance of nn.Module, got {type(module)}"
        assert isinstance(num_concurrent, int), f"num_concurrent must be an instance of int, got {type(num_concurrent)}"
        assert isinstance(max_concurrent, int), f"max_concurrent must be an instance of int, got {type(max_concurrent)}"

        self.module = module
        self.num_concurrent = num_concurrent
        self.max_concurrent = max_concurrent
        self.scheduler = get_scheduler(scheduler)
        self.parallel_context = parallel_context

    def parallelize(self) -> nn.Module:
        # TODO: wrap the model with a pipeline parallel model
        pass

    def forward(self, batches):
        partitions = None

        len(batches)
        len(partitions)

        with spawn_workers():
            pass
