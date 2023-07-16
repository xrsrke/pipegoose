from abc import ABC, abstractclassmethod
from typing import List

import torch
from torch import nn


class BasePartitioner(ABC):
    @abstractclassmethod
    def split(self):
        raise NotImplementedError


class TimePartritioner(BasePartitioner):
    def __init__(self, model: nn.Module, devices: List[torch.device]):
        self.model = model

    def split(self, data):
        for layer in self.model.named_children():
            pass
