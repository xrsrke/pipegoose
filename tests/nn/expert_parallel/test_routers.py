import torch


def test_top1_router():
    BATCH_SIZE, SEQ_LEN, D_MODEL = 5, 10, 64
    torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)

    # NOTE: dispatching_order.shape = [batch_size*seq_ken]
