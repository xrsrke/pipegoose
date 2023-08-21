import pytest
import torch
from torch import nn

from pipegoose.nn.pipeline_parallel.gpipe import GPipe


@pytest.mark.skip
def test_gpipe():
    layer1 = nn.Linear(10, 10)
    layer2 = nn.Linear(10, 10)
    model = nn.Sequential(layer1, layer2)
    input = torch.randn(10, 10, requires_grad=True)

    model = GPipe(model, balances=[1, 1], n_partitions=2)
    output = model(input)
    loss = output.sum()
    loss.backward()

    assert all([p.grad is not None for p in model.parameters()])
    assert input.grad is not None
