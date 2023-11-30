from typing import Callable

from torchtyping import TensorType

from pipegoose.nn.expert_parallel.expert_context import ExpertContext


class ExpertLoss:
    def __init__(self, loss_func: Callable, aux_weight: float = 0.01, z_weight: float = 0.1):
        self.loss_func = loss_func
        self.aux_weight = aux_weight
        self.z_weight = z_weight

    def __call__(self, *args, **kwargs) -> TensorType:
        loss = self.loss_func(*args, **kwargs)
        expert_context = ExpertContext.get_instance()
        loss += self.aux_weight * sum(expert_context.pop_all_aux_loss())
        loss += self.z_weight * sum(expert_context.pop_all_z_loss())
        return loss
