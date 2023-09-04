import torch
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.tensor_parallel.parallelize import (
    ParallelizeEmbedding,
    ParallelizeLayerNorm,
    ParallelizeLinear,
)


class TensorParallel:
    """Turn a 🤗 transformers model into a tensor parallel model."""

    def __init__(self, module: nn.Module, parallel_context: ParallelContext):
        super().__init__()
        self.module = module
        self.parallel_context = parallel_context
        self.parallelers = {
            nn.Embedding: ParallelizeEmbedding,
            nn.Linear: ParallelizeLinear,
            nn.LayerNorm: ParallelizeLayerNorm,
        }

    @torch.no_grad()
    def parallelize(self) -> nn.Module:
        for module_name, module in self.module.named_modules():
            parallelizer = self._find_parallelizer(module)
            if parallelizer is not None:
                parallelizer(module_name, module, self.parallel_context).parallelize()

        return self.module

    def _find_parallelizer(self, module):
        for module_cls, parallelizer in self.parallelers.items():
            if isinstance(module, module_cls):
                return parallelizer
        return None

    @torch.no_grad()
    def deparallelize(self) -> nn.Module:
        for module_name, module in self.module.named_modules():
            self.parallelers[module].deparallelize(module_name, module, self.parallel_context)
