from enum import Enum, auto
from typing import Any

import torch
import torch.distributed as dist

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode


class DataType(Enum):
    torch.bfloat16 = auto()
    torch.float16 = auto()
    torch.float32 = auto()
    torch.float64 = auto()

    torch.uint8 = auto()
    torch.int8 = auto()
    torch.int16 = auto()
    torch.int32 = auto()
    torch.int64 = auto()

    torch.bool = auto()


class _P2P:
    """
    Inpsired from OSLO
    Their implementation is just so good.
    """

    def _send_metadata(
        self,
        data: torch.Tensor,
        dst_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ):
        assert isinstance(data, torch.Tensor), "data must be a torch.Tensor"

        group = parallel_context.get_group(parallel_mode)

        dtype = torch.tensor(DataType[data.dtype], dtype=torch.int8)
        dist.send(dtype, dst=dst_rank, group=group)

        requires_grad = torch.tensor(1 if data.requires_grad else 0, dtype=torch.int8)
        dist.send(requires_grad, dst=dst_rank, group=group)

        shape = torch.tensor(list(data.shape), dtype=torch.long)
        dist.send(shape, dst=dst_rank, group=group)

    def _send_tensor(
        self,
        data: torch.Tensor,
        dst_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ):
        assert isinstance(data, torch.Tensor), "data must be a torch.Tensor"

        self._send_metadata(data, dst_rank, parallel_context, parallel_mode)

        group = parallel_context.get_group(parallel_mode)
        dist.send(data, dst=dst_rank, group=group)

    def _recv_tensor(
        self, src_rank: int, parallel_context: ParallelContext, parallel_mode: ParallelMode = ParallelMode.PIPELINE
    ):
        pass

    def send(
        self, data: Any, dst_rank: int, parallel_context: ParallelContext, parallel_mode: ParallelMode = ParallelMode.PIPELINE
    ):
        pass

    def recv(self, src_rank: int, parallel_context: ParallelContext, parallel_mode: ParallelMode = ParallelMode.PIPELINE):
        pass


def send(
    data: Any,
    src_rank: int,
    dst_rank: int,
    parallel_context: ParallelContext,
    parallel_mode: ParallelMode = ParallelMode.PIPELINE,
):
    if src_rank == parallel_context.get_local_rank(parallel_mode):
        _P2P().send(data, dst_rank, parallel_context, parallel_mode)


def recv(src_rank: int, dst_rank: int, parallel_context: ParallelContext, parallel_mode: ParallelMode = ParallelMode.PIPELINE):
    if dst_rank == parallel_context.get_local_rank(parallel_mode):
        _P2P().recv(src_rank, parallel_context, parallel_mode)
