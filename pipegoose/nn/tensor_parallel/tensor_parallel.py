from typing import List, Tuple, Optional

import torch
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.tensor_parallel.parallel_mapping import ParallelMapping
from pipegoose.nn.tensor_parallel.parallelize import (
    ParallelizeModule,
    ParallelizeEmbedding,
    ParallelizeLayerNorm,
    ParallelizeLinear,
    ParallelizeLMHead
)


class TensorParallel:
    """Turn a 🤗 transformers model into a tensor parallel model."""

    PARALLEL_MAPPING = [
        (lambda name, module: isinstance(module, nn.Embedding), ParallelizeEmbedding),
        (lambda name, module: isinstance(module, nn.Linear) and \
            ParallelMapping.is_lm_head(name) is False, ParallelizeLinear),
        (lambda name, module: isinstance(module, nn.Linear) and \
            ParallelMapping.is_lm_head(name) is True, ParallelizeLMHead),
        (lambda name, module: isinstance(module, nn.LayerNorm), ParallelizeLayerNorm),
    ]

    def __init__(self, module: nn.Module, parallel_context: ParallelContext):
        self.module = module
        self.parallel_context = parallel_context

    @torch.no_grad()
    def parallelize(self) -> nn.Module:
        # NOTE: because module.named_modules returns a leaf more than once,
        # this could potentially lead to the weight of a module being split
        # multiple times. so we filter out and retain the non-repetitive modules (leaf modules)
        leaf_modules = self._get_leaf_modules(self.module)
        for module_name, leaf_module in leaf_modules:
            parallelizer = self._find_parallelizer(module_name, leaf_module)
            if parallelizer is not None:
                parallelizer(module_name, leaf_module, self.module, self.parallel_context).parallelize()

        return self.module

    def _get_leaf_modules(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        leaf_modules = []
        for name, module in model.named_modules():
            if list(module.children()):
                continue
            leaf_modules.append((name, module))

        return leaf_modules

    def _find_parallelizer(self, module_name: str, module: nn.Module) -> Optional[ParallelizeModule]:
        for condition, parallelizer in self.PARALLEL_MAPPING:
            if condition(module_name, module):
                return parallelizer

    @torch.no_grad()
    def deparallelize(self) -> nn.Module:
        for module_name, module in self.module.named_modules():
            self.parallelers[module].deparallelize(module_name, module, self.parallel_context)

    def from_pretrained(self):
        pass
