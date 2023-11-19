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

class _FusedBiasGelu(torch.autograd.Function):
    """Fused gelu + bias addition function."""

    @staticmethod
    def forward(ctx, input):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass

class _FusedBiasDropout(torch.autograd.Function):
    """Fused bias + dropout function."""

    @staticmethod
    def forward(ctx, input):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass

def fused_bias_gelu(x):
    pass

def fused_bias_dropout():
    pass