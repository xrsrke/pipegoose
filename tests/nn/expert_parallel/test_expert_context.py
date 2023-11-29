from pipegoose.nn.expert_parallel.expert_context import ExpertContext


def test_expert_context():
    expert_context = ExpertContext.get_instance()

    expert_context.push_aux_loss(1.01)
    expert_context.push_z_loss(2.01)

    expert_context.push_aux_loss(1.02)
    expert_context.push_z_loss(2.02)

    # make sure that we have a singleton!
    expert_context = ExpertContext.get_instance()

    assert expert_context.pop_all_aux_loss() == [1.01, 1.02]
    assert expert_context.pop_all_aux_loss() == []

    assert expert_context.pop_all_z_loss() == [2.01, 2.02]
    assert expert_context.pop_all_z_loss() == []
