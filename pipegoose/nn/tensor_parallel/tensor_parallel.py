from typing import List, Optional, Tuple

import torch
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.expert_parallel.layers import ExpertLayer
from pipegoose.nn.parallel import Parallel
from pipegoose.nn.tensor_parallel.parallelizer import (
    EmbeddingParallelizer,
    LayerNormParallelizer,
    LinearParallelizer,
    LMHeadParallelizer,
    ModuleParallelizer,
)


class TensorParallel(Parallel):
    """Turn a 🤗 transformers model into a tensor parallel model."""

    PARALLELIZERS = [EmbeddingParallelizer, LinearParallelizer, LayerNormParallelizer, LMHeadParallelizer]

    def __init__(self, module: nn.Module, parallel_context: ParallelContext):
        self.module = module
        self.parallel_context = parallel_context

    @torch.no_grad()
    def parallelize(self) -> nn.Module:
        module = self.module

        if self.parallel_context.tensor_parallel_size > 1:
            # NOTE: because module.named_modules returns a leaf more than once,
            # this could potentially lead to the weight of a module being split
            # multiple times. so we filter out and retain the non-repetitive modules (leaf modules)
            leaf_modules = self._get_leaf_modules(module)
            for module_name, leaf_module in leaf_modules:
                parallelizer = self._find_parallelizer(module_name, leaf_module)
                if parallelizer is not None:
                    parallelizer(module_name, leaf_module, module, self.parallel_context).parallelize()

            self._save_metadata(module, self.parallel_context)

        return module

    def _get_leaf_modules(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        """Return non-expert leaf modules."""
        leaf_modules = []
        expert_names = []

        def is_child_of_expert(module_name):
            # NOTE: suppose an mlp expert has name "transformer.h.0.mlp"
            # then its children will have names like "transformer.h.0.mlp.{child_name}"
            # so we can check if a module is a child of an expert by checking if its name
            # starts with "transformer.h.0.mlp"
            for expert_name in expert_names:
                if module_name.startswith(expert_name):
                    return True
            return False

        for module_name, module in model.named_modules():
            if isinstance(module, ExpertLayer):
                expert_names.append(module_name)
                continue

            # NOTE: skip leaf modules that belong to ExpertLayer
            if is_child_of_expert(module_name) or list(module.children()):
                continue

            leaf_modules.append((module_name, module))

        return leaf_modules

    def _find_parallelizer(self, module_name: str, module: nn.Module) -> Optional[ModuleParallelizer]:
        for parallelizer in self.PARALLELIZERS:
            if parallelizer.is_parallelizable(module_name, module):
                return parallelizer
        return None

    @torch.no_grad()
    def deparallelize(self) -> nn.Module:
        for module_name, module in self.module.named_modules():
            self.PARALLELIZERS[module].deparallelize(module_name, module, self.parallel_context)
