from typing import List, Optional, Tuple

import torch
from torch import nn, fx

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.parallel import Parallel
from pipegoose.nn import FusedLayer
from pipegoose.nn.tensor_parallel.parallelizer import (
    EmbeddingParallelizer,
    LayerNormParallelizer,
    LinearParallelizer,
    LMHeadParallelizer,
    ModuleParallelizer,
)


class TensorParallel(Parallel):
    """Turn a 🤗 transformers model into a tensor parallel model."""

    PARALLELIZERS = [
        EmbeddingParallelizer,
        LinearParallelizer,
        LayerNormParallelizer,
        LMHeadParallelizer,
    ]

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
                    parallelizer(
                        module_name, leaf_module, module, self.parallel_context
                    ).parallelize()

            self._save_metadata(module, self.parallel_context)

        return module

    def _get_leaf_modules(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        leaf_modules = []
        for module_name, module in model.named_modules():
            if list(module.children()):
                continue
            leaf_modules.append((module_name, module))

        return leaf_modules

    def _find_parallelizer(
        self, module_name: str, module: nn.Module
    ) -> Optional[ModuleParallelizer]:
        for parallelizer in self.PARALLELIZERS:
            if parallelizer.is_parallelizable(module_name, module):
                return parallelizer
        return None

    @torch.no_grad()
    def deparallelize(self) -> nn.Module:
        for module_name, module in self.module.named_modules():
            self.PARALLELIZERS[module].deparallelize(
                module_name, module, self.parallel_context
            )

    def fuse(self, fused_layers: List[FusedLayer]) -> None:
        """
        In place fusion of the model's layers according to list of input layers defined in pipegoose.nn.fusion
        """
        fx_model = fx.symbolic_trace(self.module)
        modules = dict(fx_model.named_modules())

        for node in fx_model.graph.nodes:
            if node.op != "call_module":
                continue

            for fused_layer in fused_layers:
                if type(modules[node.target]) is fused_layer.represents:
                    # Replace the layer with its fused counterpart
                    *parent, name = node.target.rsplit(".", 1)
                    setattr(modules[(parent[0] if parent else '')], name, fused_layer)

        fx_model.graph.lint()
        fx_model.recompile
        return fx_model