from typing import Callable

import torch


class ExpertLoss:
    def __init__(self, loss: Callable, aux_weight: float):
        self.loss = loss
        self.aux_weight = aux_weight

    def __call__(self) -> torch.Tensor:
        pass
