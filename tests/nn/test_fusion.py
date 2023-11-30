import pytest
import torch
from torch.testing import assert_close
from torch.nn import Dropout, GELU

from pipegoose.nn.fusion import FusedBiasDropout, FusedBiasGelu


def test_FusedBiasDropout():
    dropout_p, inplace, training = 0.5, False, True
    torch.manual_seed(0)
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
    assert actual.size() == expected.size()
    assert_close(actual, expected)
    assert fused_bias_dropout.represents == Dropout
    torch.manual_seed(initial_seed)


def test_FusedBiasGelu():
    torch.manual_seed(0)
    input = torch.randn(20, 16)
    bias = torch.randn(16)

    # Find mean of application to 5 separate inputs
    actual = torch.manual_seed(0) and FusedBiasGelu.apply(input, bias)
    expected = torch.manual_seed(0) and GELU().forward(input + bias) 


    assert actual.size() == expected.size()
    assert_close(actual, expected, rtol=0.0001, atol=0.001)
    assert FusedBiasGelu.represents == GELU
