from __future__ import annotations
from typing import List

from torchtyping import TensorType


class ExpertContext:
    _instance = None
    
    def __init__(self):
        self.aux_loss = []
        self.z_loss = []

    def push_aux_loss(self, aux_loss: TensorType):
        self.aux_loss.append(aux_loss)

    def pop_all_aux_loss(self) -> List[TensorType]:
        aux_loss, self.aux_loss = self.aux_loss, []
        return aux_loss

    def push_z_loss(self, z_loss: TensorType):
        self.z_loss.append(z_loss)

    def pop_all_z_loss(self) -> List[TensorType]:
        z_loss, self.z_loss = self.z_loss, []
        return z_loss

    @classmethod
    def get_instance(cls) -> ExpertContext:
        if not cls._instance:
            cls._instance = ExpertContext()
        return cls._instance
