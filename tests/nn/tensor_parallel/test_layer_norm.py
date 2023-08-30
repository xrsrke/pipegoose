from copy import deepcopy

import pytest
import torch
from torch import nn

from pipegoose.nn.tensor_parallel.layer_norm import LayerNorm


@pytest.mark.parametrize(
    "hidden_size, normalized_shape",
    [
        (20, 20),
        (20, (20,)),
    ],
)
def test_layer_norm(hidden_size, normalized_shape):
    BATCH_SIZE = 5
    SEQ_LEN = 10

    EPS = 1e-5
    LR = 1e-3

    input = torch.randn(BATCH_SIZE, SEQ_LEN, hidden_size, requires_grad=True)

    layer_norm = nn.LayerNorm(normalized_shape=normalized_shape, eps=EPS)

    # since we assign the weight and bias to the parallel layer norm
    # we do deepcopy to make sure if the parallel layer norm do backward pass
    # it won't affect the original layer norm's weight and bias
    weight = deepcopy(layer_norm.weight)
    bias = deepcopy(layer_norm.bias)
    output = layer_norm(input)

    p_layer_norm = LayerNorm(normalized_shape=normalized_shape)
    p_layer_norm.weight = weight
    p_layer_norm.bias = bias

    p_output = p_layer_norm(input)

    assert torch.allclose(output, p_output)

    optim = torch.optim.Adam(layer_norm.parameters(), lr=LR)
    optim.zero_grad()
    output.sum().backward()
    optim.step()

    p_optim = torch.optim.Adam(p_layer_norm.parameters(), lr=LR)
    p_optim.zero_grad()
    p_output.sum().backward()
    p_optim.step()

    assert torch.allclose(p_layer_norm.weight.grad, layer_norm.weight.grad)
    assert torch.allclose(p_layer_norm.bias.grad, layer_norm.bias.grad)
