import pytest
import torch
from torch import nn

from pipegoose.nn.tensor_parallel import ParallelizeLinear


class FakeParallelContext:
    def get_world_size(self):
        return 4

    def get_rank(self):
        return 1


@pytest.fixture
def parallel_context():
    return FakeParallelContext()


def test_parallelize_linear(parallel_context):
    parallel_context.get_world_size()
    input = torch.randn(5, 10)
    linear = nn.Module(10, 20)
    output = linear(input)
    # w = deepcopy(linear.weight.data)
    # b = deepcopy(linear.bias.data)

    parallel_linear = ParallelizeLinear(linear, parallel_context).parallelize()

    assert isinstance(parallel_linear, nn.Linear)

    # TODO: assert weights[rank]
    # assert parallel_linear.module.weight.data ==

    parallel_output = parallel_linear(input)

    assert torch.allclose(output, parallel_output)

    parallel_loss = parallel_output.sum().backward()
    loss = output.sum().backward()

    assert torch.allclose(parallel_loss, loss)

    # TODO: assert grads[rank]


def test_deparallelize_linear():
    linear = nn.Linear(10, 20)

    parallel_linear = ParallelizeLinear(linear, parallel_context).parallelize()
    deparallelized_linear = parallel_linear.deparallelize()

    assert isinstance(deparallelized_linear, nn.Linear)
    assert deparallelized_linear.weight.data == linear.weight.data
    assert deparallelized_linear.bias.data == linear.bias.data
