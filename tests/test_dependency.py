import torch

from pipegoose.dependency import create_backward_dependency


def test_create_dependency():
    # define batch2 before batch1, make Backward(2) before Backward(1)
    batch2 = torch.randn(1, requires_grad=True)
    batch1 = torch.randn(1, requires_grad=True)
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

    with torch.enable_grad():
        new_batch1, new_batch2 = create_backward_dependency(source_tensor=batch1, target_tensor=batch2)

    assert id(batch1) != id(new_batch1)
    assert id(batch2) != id(new_batch2)

    # CASE 1
    # output2 = Operation.apply(2, new_batch2)
    # output1 = Operation.apply(1, new_batch1)

    # CASE 2
    output2 = Operation.apply(2, batch2)
    output1 = Operation.apply(1, batch1)

    (output2 + output1).mean().backward()

    assert timeline == [2, 1]
    assert batch1.grad is not None
    assert batch2.grad is not None


def test_fork_join_enable_grad():
    from pipegoose.dependency import EndDependency, StartDependency

    x = torch.rand(1, requires_grad=True)

    with torch.enable_grad():
        x2, p = StartDependency.apply(x)

    assert p.requires_grad
    assert x2 is not x
    x = x2

    assert x.requires_grad
    assert p.requires_grad
    assert x.grad_fn.__class__ is StartDependency._backward_cls
    assert p.grad_fn.__class__ is StartDependency._backward_cls

    with torch.enable_grad():
        x2 = EndDependency.apply(x, p)

    assert x2 is not x
    x = x2

    assert x.requires_grad
    assert x.grad_fn.__class__ is EndDependency._backward_cls


def test_create_dependency_one():
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

    # CASE 1
    # output2 = Operation.apply(2, new_batch2)
    # output1 = Operation.apply(1, new_batch1)

    # CASE 2
    output2 = Operation.apply(2, batch2)
    output1 = Operation.apply(1, batch1)

    with torch.enable_grad():
        # source_tensor: the one that execute after
        # new_output1, new_output2 = create_backward_dependency(source_tensor=output1, target_tensor=output2)
        new_output2, new_output1 = create_backward_dependency(source_tensor=output2, target_tensor=output1)

    # assert id(batch1) != id(new_batch1)
    # assert id(batch2) != id(new_batch2)

    # CASE 1
    # (output2 + output1).mean().backward()

    # CASE 2
    # (new_output2 + new_output1).mean().backward()
    (output2 + output1).mean().backward()

    assert timeline == [2, 1]
    assert batch1.grad is not None
    assert batch2.grad is not None
