from typing import Any

import torch
import torch.distributed as dist

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode

ID_TO_DTYPE = [
    torch.bfloat16,
    torch.float16,
    torch.float32,
    torch.float64,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
]

DTYPE_TO_ID = {dtype: idx for idx, dtype in enumerate(ID_TO_DTYPE)}


class _P2P:
    def __init__(self):
        self._INSTRUCTIONS = {
            torch.Tensor: {"send": self._send_tensor, "recv": self._recv_tensor},
        }

    def _send_metadata(
        self,
        data: torch.Tensor,
        dst_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ):
        assert isinstance(data, torch.Tensor), "data must be a torch.Tensor"

        group = parallel_context.get_group(parallel_mode)

        dtype = torch.tensor(DTYPE_TO_ID[data.dtype])
        dist.send(dtype, dst=dst_rank, group=group)

        requires_grad = torch.tensor(1 if data.requires_grad else 0)
        dist.send(requires_grad, dst=dst_rank, group=group)

        shape = torch.tensor(list(data.shape))
        dist.send(shape, dst=dst_rank, group=group)

    def _recv_metadata(
        self,
        src_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ):
        group = parallel_context.get_group(parallel_mode)

        dtype = torch.tensor(0)
        dist.recv(dtype, src=src_rank, group=group)
        dtype = ID_TO_DTYPE[dtype.item()]

        requires_grad = torch.tensor(0)
        dist.recv(requires_grad, src=src_rank, group=group)
        requires_grad = True if requires_grad == 1 else False

        shape = torch.tensor(0)
        dist.recv(shape, src=src_rank, group=group)
        if isinstance(shape.tolist(), int):
            shape = (shape.item(),)
        else:
            shape = tuple(shape.tolist())

        return dtype, requires_grad, shape

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
    ) -> torch.Tensor:
        group = parallel_context.get_group(parallel_mode)

        dtype, requires_grad, shape = self._recv_metadata(src_rank, parallel_context, parallel_mode)

        data = torch.zeros(size=shape, dtype=dtype, requires_grad=requires_grad)
        dist.recv(data, src=src_rank, group=group)

        return data

    def send(
        self, data: Any, dst_rank: int, parallel_context: ParallelContext, parallel_mode: ParallelMode = ParallelMode.PIPELINE
    ):
        _type = type(data)
        assert _type in self._INSTRUCTIONS, f"Type {_type} is not supported"

        self._INSTRUCTIONS[_type]["send"](data, dst_rank, parallel_context, parallel_mode)

    def recv(
        self, src_rank: int, parallel_context: ParallelContext, parallel_mode: ParallelMode = ParallelMode.PIPELINE
    ) -> torch.Tensor:
        # TODO: Add support for other types
        return self._INSTRUCTIONS[torch.Tensor]["recv"](src_rank, parallel_context, parallel_mode)
