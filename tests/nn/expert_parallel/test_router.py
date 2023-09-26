import pytest
import torch

from pipegoose.nn.expert_parallel.routers import RouterType, get_router


@pytest.mark.skip
@pytest.mark.parametrize("router_type", [RouterType.TOP_1])
def test_topk_router(router_type):
    SEQ_LEN = 10
    HIDDEN_DIM = 64
    N_EXPERTS = 5

    inputs = torch.randn(SEQ_LEN, HIDDEN_DIM)
    router = get_router(router_type)(
        n_experts=N_EXPERTS,
    )

    outputs = router(inputs)

    assert isinstance(outputs, torch.Tensor)
