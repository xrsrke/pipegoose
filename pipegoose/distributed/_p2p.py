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
    """
    P2P Communication

    NOTE: Inspired from OSLO's P2P APIs
    https://github.com/EleutherAI/oslo/blob/d7c4e32e766a99cc9d56533bc090570360dc8b2a/oslo/torch/distributed/nn/_p2p.py#L62
    """

    def __init__(self):
        self._INSTRUCTIONS = {
            torch.Tensor: {"send": self._send_tensor, "recv": self._recv_tensor},
        }

    def _send_metadata(
        self,
        data: torch.Tensor,
        dst: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode,
    ):
        assert isinstance(data, torch.Tensor), "data must be a torch.Tensor"

        group = parallel_context.get_group(parallel_mode)

        dtype = torch.tensor(DTYPE_TO_ID[data.dtype])
        dist.send(dtype, dst=dst, group=group)

        requires_grad = torch.tensor(1 if data.requires_grad else 0)
        dist.send(requires_grad, dst=dst, group=group)

        shape = torch.tensor(list(data.shape))
        dist.send(shape, dst=dst, group=group)

    def _recv_metadata(
        self,
        src: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode,
    ):
        group = parallel_context.get_group(parallel_mode)

        dtype = torch.tensor(0)
        dist.recv(dtype, src=src, group=group)
        dtype = ID_TO_DTYPE[dtype.item()]

        requires_grad = torch.tensor(0)
        dist.recv(requires_grad, src=src, group=group)
        requires_grad = True if requires_grad == 1 else False

        shape = torch.tensor(0)
        dist.recv(shape, src=src, group=group)
        if isinstance(shape.tolist(), int):
            shape = (shape.item(),)
        else:
            shape = tuple(shape.tolist())

        return dtype, requires_grad, shape

    def _send_tensor(
        self,
        data: torch.Tensor,
        dst: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode,
    ):
        assert isinstance(data, torch.Tensor), "data must be a torch.Tensor"

        self._send_metadata(data, dst, parallel_context, parallel_mode)

        group = parallel_context.get_group(parallel_mode)
        dist.send(data, dst=dst, group=group)

    def _recv_tensor(self, src: int, parallel_context: ParallelContext, parallel_mode: ParallelMode) -> torch.Tensor:
        group = parallel_context.get_group(parallel_mode)

        dtype, requires_grad, shape = self._recv_metadata(src, parallel_context, parallel_mode)

        data = torch.zeros(size=shape, dtype=dtype, requires_grad=requires_grad)
        dist.recv(data, src=src, group=group)

        return data

    def send(self, data: Any, dst: int, parallel_context: ParallelContext, parallel_mode: ParallelMode):
        _type = type(data)
        assert _type in self._INSTRUCTIONS, f"Type {_type} is not supported"

        self._INSTRUCTIONS[_type]["send"](data, dst, parallel_context, parallel_mode)

    def recv(self, src: int, parallel_context: ParallelContext, parallel_mode: ParallelMode) -> torch.Tensor:
        # TODO: Add support for other types
        return self._INSTRUCTIONS[torch.Tensor]["recv"](src, parallel_context, parallel_mode)
