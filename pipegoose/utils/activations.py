import torch
from torch import Tensor
from torch.nn import functional as F

""" 
From Issue Kernel Fusion using torch.jit #10 - https://github.com/xrsrke/pipegoose/issues/10

Fuse some popular functions and automatically replace modules in an existing 
🤗 transformers model with their corresponding fusion module

Some decisions that need to be made:
    1. Where should this be implemented?
    2. How should the automatic replacement be done?  
"""


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


class _FusedBiasGelu(torch.autograd.Function):
    """Fused gelu + bias addition function."""

    @staticmethod
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return _fused_bias_gelu_fwd(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        return (x := _fused_bias_gelu_bwd(grad_output, input, bias)), x


def fused_bias_gelu(x):
    return _FusedBiasGelu.apply(x)


@torch.jit.script
def fused_bias_dropout_train(input: Tensor, bias: Tensor, dropout_prob: float) -> Tensor:
     # type: (Tensor, Tensor, float) -> Tensor
    return F.dropout(input + bias, p=dropout_prob, training=True)


@torch.jit.script
def fused_bias_dropout_eval(input: Tensor, bias: Tensor, dropout_prob: float) -> Tensor:
     # type: (Tensor, Tensor, float) -> Tensor
    return F.dropout(input + bias, p=dropout_prob, training=False)
