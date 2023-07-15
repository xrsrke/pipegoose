from typing import List

import torch
from torch import nn


class BasePartitioner:
    def __init__(self, model: nn.Module, devices: List[torch.device]):
        self.model = model

    def __call__(self, data):
        raise NotImplementedError
