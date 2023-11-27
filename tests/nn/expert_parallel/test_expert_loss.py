from torch import nn

from pipegoose.nn.expert_parallel import ExpertLoss


def test_expert_loss():
    loss_func = nn.CrossEntropyLoss()

    expert_loss = ExpertLoss(loss_func, aux_weight=0.1)

    assert expert_loss.aux_weight == 0.1
    assert expert_loss.loss_func == loss_func

    ExpertLoss.add_aux_loss(1.01)
    ExpertLoss.add_z_loss(2.01)

    assert expert_loss.get_aux_loss() == [1.01]
    assert expert_loss.get_z_loss() == [2.01]

    ExpertLoss.add_aux_loss(1.02)
    ExpertLoss.add_z_loss(2.02)

    assert expert_loss.get_aux_loss() == [1.01, 1.02]
    assert expert_loss.get_z_loss() == [2.01, 2.02]
