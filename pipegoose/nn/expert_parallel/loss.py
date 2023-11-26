from typing import Callable
from torchtyping import TensorType

from pipegoose.nn.expert_parallel.expert_context import ExpertContext


class ExpertLoss:
    def __init__(self, loss_func: Callable, aux_weight: float, z_weight: float):
        self.loss_func = loss_func
        self.aux_weight = aux_weight
        self.z_weight = z_weight
        self._expert_context = ExpertContext()

    @property
    def expert_context(self) -> ExpertContext:
        return self._expert_context

    def __call__(self, *args, **kwargs) -> TensorType:
        loss = self.loss_func(*args, **kwargs)
        loss += self.aux_weight * sum(self._expert_context.pop_all_aux_loss())
        loss += self.z_weight * sum(self._expert_context.pop_all_z_loss())
        return loss
