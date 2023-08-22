from typing import Any, Tuple

import torch
from torch.autograd import Function

from pipegoose.distributed.functional import all_gather, all_reduce, scatter
from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode


class _Broadcast(Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, parallel_context: ParallelContext) -> torch.Tensor:
        ctx.parallel_context = parallel_context

        return tensor

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        parallel_context = ctx.parallel_context

        all_reduce(grad, parallel_context=parallel_context, parallel_mode=ParallelMode.TENSOR)

        return (grad, None, None)


class _Gather(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, dim: int, parallel_context: ParallelContext) -> torch.Tensor:
        ctx.dim = dim
        ctx.parallel_context = parallel_context

        return all_gather(input, dim=dim, async_op=False, parallel_context=parallel_context, parallel_mode=ParallelMode.TENSOR)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        dim = ctx.dim
        parallel_context = ctx.parallel_context

        return (
            scatter(grad, dim=dim, parallel_context=parallel_context, parallel_mode=ParallelMode.TENSOR),
            None,
            None,
        )


class _Scatter(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, dim: int, parallel_context: ParallelContext) -> torch.Tensor:
        ctx.dim = dim
        ctx.parallel_context = parallel_context
        return scatter(input, dim=dim, parallel_context=parallel_context, parallel_mode=ParallelMode.TENSOR)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        dim = ctx.dim
        parallel_context = ctx.parallel_context

        return (
            all_gather(
                grad_output, dim=dim, async_op=False, parallel_context=parallel_context, parallel_mode=ParallelMode.TENSOR
            ),
            None,
            None,
        )


class _Reduce(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, parallel_context: ParallelContext) -> torch.Tensor:
        return all_reduce(input, parallel_context=parallel_context, parallel_mode=ParallelMode.TENSOR)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return (grad, None)


def broadcast_to_tensor_group(input: torch.Tensor, parallel_context: ParallelContext):
    return _Broadcast.apply(input, parallel_context)


def gather_to_tensor_group(input: torch.Tensor, dim: int, parallel_context: ParallelContext):
    return _Gather.apply(input, dim, parallel_context)


def scatter_to_tensor_group(input, dim, parallel_context):
    return _Scatter.apply(input, dim, parallel_context)


def reduce_to_tensor_group(input, parallel_context):
    return _Reduce.apply(input, parallel_context)
