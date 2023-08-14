from typing import Any

import torch
from torch.autograd import Function

from pipegoose.distributed.functional import all_gather, all_reduce, scatter
from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode


class _Broadcast(Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, parallel_context: ParallelContext):
        ctx.parallel_context = parallel_context

        return tensor

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):
        parallel_context = ctx.parallel_context

        all_reduce(grad, parallel_context=parallel_context, parallel_mode=ParallelMode.TENSOR)

        return (grad, None, None)


class _Gather(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, dim: int, parallel_context: ParallelContext):
        ctx.dim = dim
        ctx.parallel_context = parallel_context

        return all_gather(input, dim=dim, async_op=False, parallel_context=parallel_context, parallel_mode=ParallelMode.TENSOR)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):
        dim = ctx.dim
        parallel_context = ctx.parallel_context

        return (
            scatter(grad, dim=dim, async_op=False, parallel_context=parallel_context, parallel_mode=ParallelMode.TENSOR),
            None,
            None,
        )


def broadcast_tensor_1d(input: torch.Tensor, parallel_context: ParallelContext):
    return _Broadcast.apply(input, parallel_context)


def gather_tensor_1d(input: torch.Tensor, dim: int, parallel_context: ParallelContext):
    return _Gather.apply(input, dim, parallel_context)
