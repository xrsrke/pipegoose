from typing import Any, Tuple

import torch


class StartDependency(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        phony = torch.empty(1, requires_grad=False, device=input.device)
        return input, phony

    @staticmethod
    def backward(ctx, grad_input: torch.Tensor, grad_phony) -> torch.Tensor:
        return grad_input


class EndDependency(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, phony: torch.Tensor) -> torch.Tensor:
        return input

    @staticmethod
    def backward(ctx: Any, grad_input) -> torch.Tensor:
        return grad_input, None


def create_backward_dependency(
    source_tensor: torch.Tensor,
    target_tensor: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    source_tensor, phony = StartDependency.apply(source_tensor)
    target_tensor = EndDependency.apply(target_tensor, phony)

    return source_tensor, target_tensor
