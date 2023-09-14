import torch
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext


class PipelineParallel:
    def __init__(self, module: nn.Module, parallel_context: ParallelContext):
        self.module = module
        self.parallel_context = parallel_context

    @torch.no_grad()
    def parallelize(self) -> nn.Module:
        self.module
