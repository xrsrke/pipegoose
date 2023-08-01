import torch

from pipegoose.distributed.functional import all_reduce
from pipegoose.distributed.parallel_mode import ParallelMode


def test_all_reduce(parallel_context):
    x = torch.randn(1, dtype=torch.float32)
    temp = x.clone()

    all_reduce(
        tensor=x,
        parallel_context=parallel_context,
        parallel_mode=ParallelMode.GLOBAL,
    )

    assert x == temp
    assert x.dtype == temp.dtype
    assert x.requires_grad == temp.requires_grad
