from abc import abstractclassmethod
from dataclasses import dataclass
from functools import partial
from typing import cast

from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode


@dataclass
class ParallelMetadata:
    is_moved_to_device: bool = False

    device: int = None
    local_device: int = None


class Parallel:
    """A base class for a parallelized module."""

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


def _to_device(self, device: str):
    """Move a parallelized module to accelerators."""
    SUPPORTED_DEVICES = ["cuda"]

    def is_specific_device(device):
        import re

        pattern = r"^cuda:[0-9]+$"
        if re.match(pattern, device):
            return True
        return False

    parallel_metadata = cast(ParallelMetadata, getattr(self, "parallel_metadata", None))

    assert parallel_metadata is not None, "Module is not parallelized yet"
    assert device in SUPPORTED_DEVICES, f"Device must be one of {SUPPORTED_DEVICES}, got {device}"
    assert parallel_metadata.is_moved_to_device is False, "Module is already moved to device"
    assert not is_specific_device(
        device
    ), f'Moving to a specific device {device} is not supported. pipegoose will handle device assignment automatically. Please use "cuda" instead'

    local_device = parallel_metadata.local_device
    for p in self.parameters():
        p = p.to(f"cuda:{local_device}")
        if p.grad is not None:
            p.grad = p.grad.to(f"cuda:{local_device}")

    for b in self.buffers():
        b = b.to(f"cuda:{local_device}")

    parallel_metadata.is_moved_to_device = True


def _to_cuda(self):
    self.to("cuda")
