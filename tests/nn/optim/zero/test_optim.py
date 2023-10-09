from copy import deepcopy

import pytest
import torch
from torch import nn, optim

from pipegoose.optim.zero import DistributedOptimizer
from pipegoose.testing.utils import init_parallel_context, spawn


def count_parameters(optimizer):
    return sum(p.numel() for group in optimizer.param_groups for p in group["params"])


def run_dist_optim(
    rank,
    world_size,
    port,
    tensor_parallel_size,
    pipeline_parallel_size,
    data_parallel_size,
    input,
    model,
    updated_model,
    optimizer,
):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    ORIG_UPDATED_MODEL = deepcopy(updated_model)
    ORIG_OPTIM = deepcopy(optimizer)
    optimizer = optim.Adam(model.parameters())
    dist_optimizer = DistributedOptimizer(optimizer, parallel_context)

    assert dist_optimizer.defaults == optimizer.defaults
    assert len(dist_optimizer.param_groups) == len(optimizer.param_groups)
    assert dist_optimizer.state_dict().keys() == optimizer.state_dict().keys()
    # NOTE: test whether the optimizer partitions the parameters across data parallel dimension
    assert count_parameters(dist_optimizer) < count_parameters(ORIG_OPTIM)

    dist_optimizer.zero_grad()
    model(input).sum().backward()
    dist_optimizer.step()

    # NOTE: test whether the model parameters are updated correctly
    for p1, p2 in zip(model.parameters(), ORIG_UPDATED_MODEL.parameters()):
        assert not torch.allclose(p1, p2), f"p1: {p1}, p2: {p2}"

    # dist_optimizer.zero_grad()

    # for p in model.parameters():
    #     assert p.grad is None


@pytest.mark.parametrize("data_parallel_size", [2, 5])
def test_dist_optim(data_parallel_size):
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 1
    WORLD_SIZE = TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE * data_parallel_size

    # LR = 1e-3
    BATCH_SIZE = 5
    HIDDEN_SIZE = 10
    OUTPUT_SIZE = 2

    input = torch.randn(BATCH_SIZE, HIDDEN_SIZE)
    model = nn.Sequential(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.ReLU(), nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE))
    ORIG_INPUT = deepcopy(input)
    ORIG_MODEL = deepcopy(model)
    optimizer = optim.Adam(model.parameters())
    optimizer.zero_grad()

    model(input).sum().backward()
    optimizer.step()

    spawn(
        run_dist_optim,
        world_size=WORLD_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=data_parallel_size,
        input=ORIG_INPUT,
        model=ORIG_MODEL,
        updated_model=model,
        optimizer=optimizer,
    )
