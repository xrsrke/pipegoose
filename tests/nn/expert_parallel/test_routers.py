import torch
import torch.nn.functional as F

from pipegoose.nn.expert_parallel import SwitchNoisePolicy, Top1Router, Top2Router


def run_topk_router(router, batch_size, seq_len, d_model, num_experts, top_k):
    router.train()

    input = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    router_output = router(input)

    assert router_output.dispatching_order.shape == (batch_size * seq_len, num_experts)
    assert router_output.weight.shape == (batch_size * seq_len, num_experts)
    assert router_output.aux_loss.shape == ()
    assert router_output.z_loss.shape == ()

    total_tokens = batch_size * seq_len

    if hasattr(router, "_expert_capacity") and router.expert_capacity:
        expert_capacity = router._expert_capacity(total_tokens)

        for expert_id in range(num_experts):
            assert router_output.dispatching_order[..., expert_id].sum().item() < expert_capacity

        for token_id in range(total_tokens):
            assert router_output.dispatching_order[token_id, ...].sum().item() <= top_k

    else:
        for token_id in range(total_tokens):
            assert router_output.dispatching_order[token_id, ...].sum().item() == top_k

    # test backward pass

    target_weight = torch.randn_like(router_output.weight)  # Random target for testing

    loss = router_output.aux_loss + router_output.z_loss
    loss += F.mse_loss(router_output.weight, target_weight)

    loss.backward()

    # check the gradients
    assert input.grad is not None, "Input gradient should not be None"
    assert not torch.all(input.grad == 0), "Input gradient should not be all zeros"
    for param in router.parameters():
        assert param.grad is not None, "Parameter gradient should not be None"
        assert not torch.all(param.grad == 0), "Parameter gradient should not be all zeros"


def test_top1_router():
    NUM_EXPERTS = 5
    BATCH_SIZE, SEQ_LEN, D_MODEL = 5, 10, 64

    noise_policy = SwitchNoisePolicy()
    top1_router = Top1Router(noise_policy, NUM_EXPERTS, D_MODEL)

    run_topk_router(top1_router, BATCH_SIZE, SEQ_LEN, D_MODEL, NUM_EXPERTS, top_k=1)


def test_top1_router_with_expert_capacity():
    NUM_EXPERTS = 5
    BATCH_SIZE, SEQ_LEN, D_MODEL = 5, 10, 64

    noise_policy = SwitchNoisePolicy()
    top1_router = Top1Router(noise_policy, NUM_EXPERTS, D_MODEL, expert_capacity=(1.0, 2.0))

    run_topk_router(top1_router, BATCH_SIZE, SEQ_LEN, D_MODEL, NUM_EXPERTS, top_k=1)


def test_top2_router():
    NUM_EXPERTS = 5
    BATCH_SIZE, SEQ_LEN, D_MODEL = 5, 10, 64

    noise_policy = SwitchNoisePolicy()
    top2_router = Top2Router(noise_policy, NUM_EXPERTS, D_MODEL)

    run_topk_router(top2_router, BATCH_SIZE, SEQ_LEN, D_MODEL, NUM_EXPERTS, top_k=2)


def test_top2_router_with_expert_capacity():
    NUM_EXPERTS = 5
    BATCH_SIZE, SEQ_LEN, D_MODEL = 5, 10, 64

    noise_policy = SwitchNoisePolicy()
    top2_router = Top2Router(noise_policy, NUM_EXPERTS, D_MODEL, expert_capacity=(1.0, 2.0))

    run_topk_router(top2_router, BATCH_SIZE, SEQ_LEN, D_MODEL, NUM_EXPERTS, top_k=2)
