import torch
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.tensor_parallel.parallelize import (
    # ParallelizeAttention,
    ParallelizeEmbedding,
    ParallelizeLayerNorm,
    ParallelizeLinear,
)


class TensorParallel:
    """Turn a transformers model into a tensor parallel model."""

    def __init__(self, module: nn.Module, parallel_context: ParallelContext):
        super().__init__()
        self.module = module
        self.parallel_context = parallel_context

    @torch.no_grad()
    def parallelize(self):
        for module_name, module in self.module.named_modules():
            if isinstance(module, nn.Embedding):
                ParallelizeEmbedding(module_name, module, self.parallel_context).parallelize()
            elif isinstance(module, nn.Linear):
                ParallelizeLinear(module_name, module, self.parallel_context).parallelize()
            elif isinstance(module, nn.LayerNorm):
                ParallelizeLayerNorm(module_name, module, self.parallel_context).parallelize()
