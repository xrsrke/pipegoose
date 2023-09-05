from copy import deepcopy

import pytest
import torch
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.tensor_parallel.layer_norm import LayerNorm
from pipegoose.testing.utils import spawn


def init_parallel_context(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    parallel_context = ParallelContext(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        host="localhost",
        port=port,
        seed=69,
        backend="gloo",
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )

    return parallel_context


def run_layer_norm(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, normalized_shape, input, output, weight, bias, weight_grad, bias_grad
):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    p_layer_norm = LayerNorm(normalized_shape=normalized_shape, parallel_context=parallel_context)
    p_layer_norm.weight.data = weight
    p_layer_norm.bias.data = bias

    p_output = p_layer_norm(input)

    assert torch.allclose(output, p_output)

    p_output.sum().backward()

    assert torch.allclose(p_layer_norm.weight.grad, weight_grad)
    assert torch.allclose(p_layer_norm.bias.grad, bias_grad)


@pytest.mark.parametrize(
    "tensor_parallel_size", [1, 2]
)
@pytest.mark.parametrize("hidden_size", [20])
@pytest.mark.parametrize("normalized_shape", [20, (20,)])
def test_layer_norm(tensor_parallel_size, hidden_size, normalized_shape):
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    BATCH_SIZE = 5
    SEQ_LEN = 10
    EPS = 1e-5

    input = torch.randn(BATCH_SIZE, SEQ_LEN, hidden_size, requires_grad=True)

    layer_norm = nn.LayerNorm(normalized_shape, eps=EPS)

    # NOTE: since we assign the weight and bias to the parallel layer norm
    # we do deepcopy to make sure if the parallel layer norm do backward pass
    # it won't affect the original layer norm's weight and bias
    weight = deepcopy(layer_norm.weight)
    bias = deepcopy(layer_norm.bias)
    output = layer_norm(input)
    output.sum().backward()

    weight_grad = deepcopy(layer_norm.weight.grad)
    bias_grad = deepcopy(layer_norm.bias.grad)

    spawn(
        run_layer_norm,
        world_size=tensor_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        normalized_shape=normalized_shape,
        input=input.detach(),
        output=output.detach(),
        weight=weight.detach(),
        bias=bias.detach(),
        weight_grad=weight_grad.detach(),
        bias_grad=bias_grad.detach()
    )
