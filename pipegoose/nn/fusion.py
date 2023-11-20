import torch
from typing import Any, Type
from torch import Tensor
from torch.nn import functional as F
from torch.nn import GELU, Dropout, Module


class FusedLayer:
    # Used to match layers in fx.GraphModule to their fused layer counterpart
    represents: Type[Module]


@torch.jit.script
def _fused_bias_gelu_fwd(input, bias):
    x = input + bias
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


@torch.jit.script
def _fused_bias_gelu_bwd(g, input, bias):
    x = input + bias
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)
    return ff * g


class FusedBiasGelu(torch.autograd.Function, FusedLayer):
    """Fused gelu + bias function."""

    represents = GELU

    @staticmethod
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return _fused_bias_gelu_fwd(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        return (x := _fused_bias_gelu_bwd(grad_output, input, bias)), x


@torch.jit.script
def fused_bias_dropout(
    input: Tensor, bias: Tensor, dropout_prob: float, training: bool
) -> Tensor:
    # type: (Tensor, Tensor, float, bool) -> Tensor
    return F.dropout(input + bias, p=dropout_prob, training=training)


class FusedBiasDropout(torch.autograd.Function, FusedLayer):
    """Fused dropout + bias function."""

    represents = Dropout

    @staticmethod
    def forward(ctx, input, bias, dropout_prob: float, training: bool):
        return fused_bias_dropout(input, bias, dropout_prob, training)

    @staticmethod
    def backward(ctx, grad_output) -> Any:
        # TODO: How should this behave during eval vs. training? And how do we detect which mode
        #  we're in given that this is a static method?
        return grad_output
