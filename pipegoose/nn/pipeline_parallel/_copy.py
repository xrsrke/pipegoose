from typing import Tuple

import torch

from pipegoose.nn.pipeline_parallel.stream import (
    StreamType,
    default_stream,
    get_device,
    record_stream,
    use_stream,
)


class Context:
    prev_stream: torch.cuda.Stream
    next_stream: torch.cuda.Stream


class Copy(torch.autograd.Function):
    """Synchronous dat transfer between streams."""

    @staticmethod
    def forward(ctx: Context, prev_stream: StreamType, next_stream: StreamType, input: torch.Tensor) -> torch.Tensor:
        ctx.prev_stream = prev_stream
        ctx.next_stream = next_stream

        compute_stream = default_stream(get_device(next_stream))

        with use_stream(prev_stream), use_stream(next_stream):
            moved_input = input.to(get_device(next_stream))
            record_stream(input, prev_stream)
            record_stream(moved_input, compute_stream)

        return moved_input

    @staticmethod
    def backward(ctx: Context, grad_input: torch.Tensor) -> Tuple[None, None, torch.Tensor]:
        prev_stream = ctx.prev_stream
        next_stream = ctx.next_stream

        compute_stream = default_stream(get_device(prev_stream))

        with use_stream(prev_stream), use_stream(next_stream):
            moved_grad_input = grad_input.to(get_device(prev_stream))

            record_stream(grad_input, next_stream)
            record_stream(moved_grad_input, compute_stream)

        return tuple([None, None, grad_input])
