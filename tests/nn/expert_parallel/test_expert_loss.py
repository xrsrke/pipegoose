import torch
from torch import nn
import torch.nn.functional as F

from pipegoose.nn.expert_parallel import ExpertLoss


def test_expert_loss():
    torch.manual_seed(42)
    logits = torch.randn((10, 5))
    gt = torch.randn((10, 5))

    loss_func = nn.MSELoss()

    expert_loss = ExpertLoss(loss_func, aux_weight=0.1, z_weight=0.2)
    expert_context = expert_loss.expert_context

    assert expert_loss.aux_weight == 0.1
    assert expert_loss.z_weight == 0.2
    assert expert_loss.loss_func == loss_func

    expert_context.push_aux_loss(1.01)
    expert_context.push_z_loss(2.01)

    expert_context.push_aux_loss(1.02)
    expert_context.push_z_loss(2.02)

    expected_loss = F.mse_loss(logits, gt) + 0.1 * (1.01 + 1.02) + 0.2 * (2.01 + 2.02)
    loss = expert_loss(logits, gt)

    assert torch.allclose(loss, expected_loss)

    assert expert_context.aux_loss == []
    assert expert_context.z_loss == []
