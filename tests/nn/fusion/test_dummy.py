import torch
import pytest
from pipegoose.nn.fusion.ops import FusedDummy

@pytest.mark.parametrize("N", [10])
def test_fused_dummy(N):
    X = torch.arange(N, dtype=torch.float32) 
    fused_op = FusedDummy().load()
    output = fused_op.forward(X)
    assert torch.allclose(output, X)