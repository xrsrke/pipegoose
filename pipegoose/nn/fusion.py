import torch
from typing import Type, Literal
from multimethod import overload
from torch import Tensor
from torch.nn import functional as F

from torch.nn import GELU, Dropout, Module
from torch.nn.modules.dropout import _DropoutNd
from transformers.models.bloom.modeling_bloom import BloomGelu



class FusedLayer:
    # Used to match layers in Parallel.module to their fused layer counterpart
    represents: list[Type[Module]]

    # We pass the target_layer to give each fused layer the ability to copy its instantiation arguments
    def __init__(self, target_layer: Module) -> None:
        pass


@torch.jit.script
def _fused_bias_gelu_fwd(input, bias):
    x = input + bias
    return x * 0.5 * (1.0 + torch.tanh(0.7978845608028654 * x * (1 + 0.044715 * x * x)))


@torch.jit.script
def _fused_bias_gelu_bwd(g, input, bias):
    x = input + bias
    tanh_out = torch.tanh(0.7978845608028654 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.7978845608028654 + 0.1070322244089 * x * x)
    ) + 0.5 * (1 + tanh_out)
    return ff * g


class FusedBiasGelu(GELU, FusedLayer):
    """Fused gelu + bias function."""

    represents = [GELU, BloomGelu]
    approximate: str

    @overload
    def __init__(self, target_layer: GELU):
        super().__init__()
        self.approximate = target_layer.approximate

    @overload
    def __init__(self, target_layer: BloomGelu):
        super().__init__()

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
    input: Tensor,
    bias: Tensor,
    dropout_prob: float,
    training: bool,
    inplace: bool = False,
) -> Tensor:
    # type: (Tensor, Tensor, float, bool, bool) -> Tensor
    return F.dropout(input + bias, p=dropout_prob, training=training, inplace=inplace)


class FusedBiasDropout(_DropoutNd, FusedLayer):
    """
    Fused dropout + bias function.
    See: https://pytorch.org/docs/stable/_modules/torch/nn/modules/dropout.html#Dropout
    """

    represents = [Dropout]

    def __init__(self, target_layer: Dropout):
        dropout_p = target_layer.p
        inplace = target_layer.inplace
        super().__init__(p=dropout_p, inplace=inplace)

    def forward(self, input: Tensor, bias: Tensor):
        return fused_bias_dropout(
            input, bias, self.p, self.training, self.inplace
        )
