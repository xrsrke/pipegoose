import torch

from pipegoose.dependency import create_backward_dependency


def test_create_dependency():
    batch1 = torch.randn(1, requires_grad=True)
    batch2 = torch.randn(1, requires_grad=True)
    timeline = []

    class Operation(torch.autograd.Function):
        @staticmethod
        def forward(ctx, number, input):
            ctx.number = number
            return input

        @staticmethod
        def backward(ctx, grad_output):
            nonlocal timeline
            timeline.append(ctx.number)
            return tuple([None, grad_output])

    new_batch1, new_batch2 = create_backward_dependency(source_tensor=batch1, target_tensor=batch2)

    assert new_batch1 == batch1
    assert new_batch2 == batch2
    assert new_batch1.requires_grad == batch1.requires_grad
    assert new_batch2.requires_grad == batch2.requires_grad

    output1 = Operation.apply(1, new_batch1)
    output2 = Operation.apply(2, new_batch2)
    (output1 + output2).backward()

    assert timeline == [2, 1]
    assert batch1.grad is not None
    assert batch2.grad is not None
