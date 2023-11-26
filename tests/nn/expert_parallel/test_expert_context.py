from pipegoose.nn.expert_parallel import ExpertContext


def test_expert_context():
    expert_context = ExpertContext()

    expert_context.push_aux_loss(1.01)
    expert_context.push_z_loss(2.01)

    expert_context.push_aux_loss(1.02)
    expert_context.push_z_loss(2.02)

    assert expert_context.pop_all_aux_loss() == [1.01, 1.02]
    assert expert_context.pop_all_aux_loss() == []
    
    assert expert_context.pop_all_z_loss() == [2.01, 2.02]
    assert expert_context.pop_all_z_loss() == []
