import torch

from pipegoose.nn.expert_parallel import Top1Router, Top2Router


def run_topk_router(
    router,
    batch_size,
    seq_len,
    d_model,
    num_experts,
    top_k
):
    input = torch.randn(batch_size, seq_len, d_model)
    dispatch_order, gate_values, loss = router(input)

    assert dispatch_order.shape == (batch_size*seq_len, num_experts)
    assert gate_values.shape == (batch_size*seq_len, num_experts)
    assert loss.shape == ()

    total_tokens = batch_size * seq_len
    expert_capacity = router._expert_capacity(total_tokens)

    for expert_id in range(num_experts):
        assert dispatch_order[..., expert_id].sum().item() < expert_capacity

    for token_id in range(total_tokens):
        assert dispatch_order[token_id, ...].sum().item() <= top_k


def test_top1_router():
    NUM_EXPERTS = 5
    BATCH_SIZE, SEQ_LEN, D_MODEL = 5, 10, 64

    top1_router = Top1Router(NUM_EXPERTS, D_MODEL)

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

    top2_router = Top2Router(NUM_EXPERTS, D_MODEL)

    run_topk_router(
        top2_router,
        BATCH_SIZE,
        SEQ_LEN,
        D_MODEL,
        NUM_EXPERTS,
        top_k=2
    )