import pytest
import torch
from torch.nn import Dropout, GELU

from pipegoose.nn.fusion import FusedBiasDropout, FusedBiasGelu


def test_FusedBiasDropout():
    dropout = FusedBiasDropout()
    input = torch.randn(20, 16)
    args = (0.5, False)
    expected = Dropout(*args)(input)
    actual = dropout(input)

    assert actual.size() == expected.size()
    assert torch.allclose(actual, expected)
    assert FusedBiasDropout(*args).represents == Dropout


def test_FusedBiasGelu():
    gelu = FusedBiasGelu()
    input = torch.randn(20, 16)
    expected = GELU()(input)
    actual = gelu(input)

    assert actual.size() == expected.size()
    assert torch.allclose(actual, expected)
    assert FusedBiasGelu().represents == GELU
