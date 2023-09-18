"""DON'T USE THIS MODULE: Under development."""

from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext


class ExpertParallel:
    def __init__(self, module: nn.Module, parallel_context: ParallelContext):
        self.module = module
        self.parallel_context = parallel_context
