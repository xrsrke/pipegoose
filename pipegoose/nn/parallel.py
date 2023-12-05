from abc import abstractclassmethod
from dataclasses import dataclass
from functools import partial
from typing import cast, List
from copy import deepcopy

import torch
from torch import nn
import torch.fx as fx

from pipegoose.nn.fusion import FusedLayer, should_fuse_layer, replace_layer_in_module
from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode


@dataclass
class ParallelMetadata:
    device: int = None
    local_device: int = None


class Parallel:
    """A base class for a parallelized module."""
    def __init__(self, module: nn.Module, parallel_context: ParallelContext):
        self.module = module
        self.parallel_context = parallel_context

    @abstractclassmethod
    def parallelize(self):
        """Parallelize the module."""
        raise NotImplementedError

    @abstractclassmethod
    def deparallelize(self):
        """Deparallelize the module."""
        raise NotImplementedError

    def _save_metadata(self, module: nn.Module, parallel_context: ParallelContext):
        def _get_device(parallel_context: ParallelContext) -> int:
            rank = parallel_context.get_global_rank()
            tp_rank = parallel_context.get_local_rank(ParallelMode.TENSOR)
            pp_rank = parallel_context.get_local_rank(ParallelMode.PIPELINE)
            dp_rank = parallel_context.get_local_rank(ParallelMode.DATA)

            ranks = (
                (ParallelMode.GLOBAL, rank),
                (ParallelMode.TENSOR, tp_rank),
                (ParallelMode.PIPELINE, pp_rank),
                (ParallelMode.DATA, dp_rank),
            )
            device = parallel_context.ranks2device(ranks)
            local_device = device % parallel_context.get_world_size(ParallelMode.GLOBAL)
            return device, local_device

        device, local_device = _get_device(parallel_context)
        parallel_metadata = ParallelMetadata(
            device=device,
            local_device=local_device,
        )
        setattr(module, "parallel_metadata", parallel_metadata)
        setattr(module, "to", partial(_to_device, module))
        setattr(module, "cuda", partial(_to_cuda, module))

    def fuse_md(self, fused_layers: List[FusedLayer]) -> nn.Module:
        """
        In place fusion of the model's layers according to list of input layers defined in pipegoose.nn.fusion
        """
        replacements: list[tuple[str, nn.Module]] = [
            (name, fused_layer(child))
            for name, child in self.module.named_modules()
            for fused_layer in fused_layers
            if any(map(lambda r: isinstance(child, r), fused_layer.represents))
        ]

        for name, replacement in replacements: setattr(self.module, name, replacement)
            
        return self.module

    def fuse(self, fused_layers: List[FusedLayer]) -> nn.Module:

        # Collect functions to wrap in the tracer
        autowrap_fns = tuple(set.union(*map(lambda l: set(l.wraps), fused_layers)))
        # The arguments to the tracer should be configured based on the union of the 
        #  FusedLayer's 'wraps' attribute, which defines the operations that their 
        #  representations contain that are not torch.jitable, such as `len` in
        #  BloomGelu
        graph = fx.Tracer(autowrap_functions=autowrap_fns).trace(self.module)
        graph_module = fx.GraphModule(self.module, graph)

        modules = dict(graph_module.named_modules())
        new_graph = deepcopy(graph_module.graph)

        print(f'--------------------- Attempting to edit: ---------------------')
        graph.print_tabular()
        # A Graph is a data structure that represents a method on a GraphModule
        for node in new_graph.nodes:
            for candidate in fused_layers:
                if should_fuse_layer(node, candidate, modules):
                    if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                        continue
                    replace_layer_in_module(node.args[0], candidate, modules)
                    # After replacing the node in-place, replace it in all other occurrences in the graph
                    node.replace_all_uses_with(node.args[0])
                    # NOTE We cannot erase the node until nobody is using it anymore
                    new_graph.erase_node(node)
                    break

        graph.lint()
        return torch.fx.GraphModule(self.module, new_graph)


def _to_device(self, device: str):
    """Move a parallelized module to accelerators."""
    SUPPORTED_DEVICES = ["cuda", "gpu"]

    def is_specific_device(device):
        import re

        pattern = r"^cuda:[0-9]+$"
        if re.match(pattern, device):
            return True
        return False

    parallel_metadata = cast(ParallelMetadata, getattr(self, "parallel_metadata", None))

    assert parallel_metadata is not None, "Module is not parallelized yet"
    assert (
        device in SUPPORTED_DEVICES
    ), f"Device must be one of {SUPPORTED_DEVICES}, got {device}"
    assert not is_specific_device(
        device
    ), f'Moving to a specific device {device} is not supported. pipegoose will handle device assignment automatically. Please use "cuda" instead'

    if torch.cuda.device_count() == 0:
        raise RuntimeError("There are no GPUs available.")

    local_device = parallel_metadata.local_device
    for p in self.parameters():
        p.data = p.to(f"cuda:{local_device}")
        if p.grad is not None:
            p.grad.data = p.grad.to(f"cuda:{local_device}")

    for b in self.buffers():
        b.data = b.to(f"cuda:{local_device}")


def _to_cuda(self):
    self.to("cuda")
