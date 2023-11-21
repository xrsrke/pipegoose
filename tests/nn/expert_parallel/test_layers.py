import pytest
import torch
from torch import nn

from pipegoose.distributed.functional import all_reduce
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.expert_parallel.layers import ExpertLayer
from pipegoose.testing.utils import count_model_parameters, init_parallel_context, spawn


class DummyRouter:
    def __init__(self, num_experts):
        self.num_experts = num_experts

    def __call__(self, inputs):
        n_tokens = inputs.shape[0] * inputs.shape[1]
        return torch.randint(0, self.num_experts, (n_tokens,)), None, None


def run_expert_layer(
    rank,
    world_size,
    port,
    tensor_parallel_size,
    pipeline_parallel_size,
    data_parallel_size,
    inputs,
    num_experts,
    expert,
    router,
    enable_tensor_parallel,
):
    parallel_context = init_parallel_context(
        rank,
        world_size,
        port,
        tensor_parallel_size,
        pipeline_parallel_size,
        data_parallel_size,
    )

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    expert_layer = ExpertLayer(
        num_experts,
        expert,
        router,
        enable_tensor_parallel,
        parallel_context,
    )

    local_param_count = count_model_parameters(expert_layer)
    total_param_count = all_reduce(
        torch.tensor(local_param_count), parallel_context=parallel_context, parallel_mode=ParallelMode.TENSOR
    )
    assert total_param_count == count_model_parameters(expert) * num_experts
    assert all(isinstance(x, type(expert)) for x in expert_layer.experts)

    outputs = expert_layer(inputs)

    assert outputs.shape == inputs.shape
    assert not (outputs == 0).all(dim=-1).any(), "There is at least one input embedding that doesn't go through any experts."


@pytest.mark.parametrize("tensor_parallel_size, num_experts", [(1, 1), (2, 2), (2, 4), (8, 8)])
@pytest.mark.parametrize("enable_tensor_parallel", [False])
def test_expert_layer(tensor_parallel_size, num_experts, enable_tensor_parallel):
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1
    WORLD_SIZE = tensor_parallel_size * PIPELINE_PARALLEL_SIZE * DATA_PARALLEL_SIZE
    BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE = 5, 10, 64

    inputs = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
    expert = nn.Sequential(
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE * 4),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE * 4, HIDDEN_SIZE),
    )
    router = DummyRouter(num_experts)

    spawn(
        run_expert_layer,
        world_size=WORLD_SIZE,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        inputs=inputs,
        num_experts=num_experts,
        expert=expert,
        router=router,
        enable_tensor_parallel=enable_tensor_parallel,
    )
