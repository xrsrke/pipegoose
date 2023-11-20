import pytest
import torch
from torch.nn import Dropout, GELU

from pipegoose.nn.fusion import FusedBiasDropout, FusedBiasGelu


def test_FusedBiasDropout():
    dropout_p, inplace, training = 0.5, False, True

    fused_bias_dropout = FusedBiasDropout(
        dropout_p=dropout_p, inplace=inplace
    )

    input = torch.randn(20, 16)
    bias = torch.randn(16)
    torch.manual_seed(0)

    expected = Dropout(p=dropout_p, inplace=inplace)(input + bias)
    actual = fused_bias_dropout(input, bias)
    
    assert actual.size() == expected.size()
    assert torch.allclose(actual, expected)
    assert fused_bias_dropout.represents == Dropout


def test_FusedBiasGelu():
    fused_bias_gelu = FusedBiasGelu()
    input = torch.randn(20, 16)
    bias = torch.randn(16)
    torch.manual_seed(0)

    expected = GELU()(input + bias)
    actual = fused_bias_gelu(input, bias)

    assert actual.size() == expected.size()
    assert torch.allclose(actual, expected)
    assert fused_bias_gelu.represents == GELU
