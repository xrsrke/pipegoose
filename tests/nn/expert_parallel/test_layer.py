import torch
from torch import nn

from pipegoose.nn.expert_parallel.layers import MoELayer
from pipegoose.nn.expert_parallel.routers import Top1Router


def test_moe_layer():
    BATCH_SIZE = 10
    SEQ_LEN = 5
    HIDDEN_DIM = 64
    N_EXPERTS = 10

    inputs = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    expert = nn.Linear(10, 10)

    router = Top1Router(n_experts=N_EXPERTS)

    layer = MoELayer(
        expert=expert,
        n_experts=10,
        router=router,
    )

    outputs = layer(inputs)

    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
