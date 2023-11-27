import torch
import torch.nn.functional as F

from pipegoose.nn.expert_parallel import Top1Router, Top2Router, SwitchNoisePolicy


def run_topk_router(
    router,
    batch_size,
    seq_len,
    d_model,
    num_experts,
    top_k
):
    router.train()

    input = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    dispatch_order, gate_values, loss = router(input)

    assert dispatch_order.shape == (batch_size*seq_len, num_experts)
    assert gate_values.shape == (batch_size*seq_len, num_experts)
    assert loss.shape == ()

    total_tokens = batch_size * seq_len

    if hasattr(router, "_expert_capacity") and router.expert_capacity:
        expert_capacity = router._expert_capacity(total_tokens)

        for expert_id in range(num_experts):
            assert dispatch_order[..., expert_id].sum().item() < expert_capacity

        for token_id in range(total_tokens):
            assert dispatch_order[token_id, ...].sum().item() <= top_k

    else:
        for token_id in range(total_tokens):
            assert dispatch_order[token_id, ...].sum().item() == top_k

    # test backwardpass

    target_gate_values = torch.randn_like(gate_values) # Random target for testing
    loss += F.mse_loss(gate_values, target_gate_values)

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

    run_topk_router(
        top1_router,
        BATCH_SIZE,
        SEQ_LEN,
        D_MODEL,
        NUM_EXPERTS,
        top_k=1
    )


def test_top1_router_with_expert_capacity():
    NUM_EXPERTS = 5
    BATCH_SIZE, SEQ_LEN, D_MODEL = 5, 10, 64

    noise_policy = SwitchNoisePolicy()
    top1_router = Top1Router(noise_policy, NUM_EXPERTS, D_MODEL, expert_capacity=(1.0, 2.0))

    run_topk_router(
        top1_router,
        BATCH_SIZE,
        SEQ_LEN,
        D_MODEL,
        NUM_EXPERTS,
        top_k=1
    )


def test_top2_router():
    NUM_EXPERTS = 5
    BATCH_SIZE, SEQ_LEN, D_MODEL = 5, 10, 64

    noise_policy = SwitchNoisePolicy()
    top2_router = Top2Router(noise_policy, NUM_EXPERTS, D_MODEL)

    run_topk_router(
        top2_router,
        BATCH_SIZE,
        SEQ_LEN,
        D_MODEL,
        NUM_EXPERTS,
        top_k=2
    )


def test_top2_router_with_expert_capacity():
    NUM_EXPERTS = 5
    BATCH_SIZE, SEQ_LEN, D_MODEL = 5, 10, 64

    noise_policy = SwitchNoisePolicy()
    top2_router = Top2Router(noise_policy, NUM_EXPERTS, D_MODEL, expert_capacity=(1.0, 2.0))

    run_topk_router(
        top2_router,
        BATCH_SIZE,
        SEQ_LEN,
        D_MODEL,
        NUM_EXPERTS,
        top_k=2
    )
