import pytest
import torch
from torch.nn import Dropout, GELU

from pipegoose.nn.fusion import FusedBiasDropout, FusedBiasGelu


def test_FusedBiasDropout():
    dropout_p, inplace, training = 0.5, False, True

    fused_bias_dropout = FusedBiasDropout(
        dropout_p=dropout_p, inplace=inplace
    )
    initial_seed = torch.initial_seed()
    input = torch.randn(20, 16)
    bias = torch.randn(16)
    torch.manual_seed(0)

    expected = Dropout(p=dropout_p, inplace=inplace)(input + bias)
    # Reset manual seed after each random operation
    torch.manual_seed(0)

    actual = fused_bias_dropout(input, bias)
    import pdb; pdb.set_trace()
    assert actual.size() == expected.size()
    assert torch.allclose(actual, expected)
    assert fused_bias_dropout.represents == Dropout
    torch.manual_seed(initial_seed)


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
