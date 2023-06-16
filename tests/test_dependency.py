import torch

from pipegoose.dependency import create_backward_dependency


def test_create_dependency():
    batch1 = torch.randn(1, requires_grad=True)
    batch2 = torch.randn(1, requires_grad=True)
    logs = []

    class Operation(torch.autograd.Function):
        @staticmethod
        def forward(ctx, number, input):
            ctx.number = number
            return input

        @staticmethod
        def backward(ctx, grad_output):
            nonlocal logs
            logs.append(ctx.number)
            return None, grad_output

    batch1, batch2 = create_backward_dependency(
        source_tensor=batch1,
        target_tensor=batch2
    )

    batch1 = Operation.apply(1, batch1)
    batch2 = Operation.apply(2, batch2)
    (batch1+batch2).backward()

    assert logs == [2, 1]
