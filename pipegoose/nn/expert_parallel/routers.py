"""DON'T USE THIS MODULE: Under development."""
from abc import ABC

from torch import nn


class Router(ABC):
    pass


class Top1Router(Router, nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class Top2Router(Router, nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass
