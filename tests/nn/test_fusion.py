import pytest
from unittest.mock import MagicMock
import torch
from torch.testing import assert_close
from torch.nn import Dropout, GELU

from pipegoose.nn.fusion import FusedBiasDropout, FusedBiasGelu


def test_FusedBiasDropout():
    dropout_p, inplace, training = 0.5, False, True
    input = torch.randn(20, 16)
    bias = torch.randn(16)
    
    # Reset manual seed after each random operation
    torch_dropout = torch.manual_seed(0) and Dropout(p=dropout_p, inplace=inplace)

    expected = torch_dropout(input + bias)

    fused_bias_dropout = FusedBiasDropout(torch_dropout)
    actual = torch.manual_seed(0) and fused_bias_dropout(input, bias)
    
    assert actual.size() == expected.size()
    assert_close(actual, expected)
    assert fused_bias_dropout.represents == [Dropout]


def test_FusedBiasGelu():
    torch.manual_seed(0)
    input = torch.randn(20, 16)
    bias = torch.randn(16)

    expected = torch.manual_seed(0) and GELU().forward(input + bias) 
    actual = torch.manual_seed(0) and FusedBiasGelu.forward(MagicMock(), input, bias)

    assert actual.size() == expected.size()
    assert_close(actual, expected, rtol=0.0001, atol=0.001)
    assert GELU in FusedBiasGelu.represents
