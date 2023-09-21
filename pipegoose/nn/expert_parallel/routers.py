"""DON'T USE THIS MODULE: Under development."""
from abc import ABC
from enum import Enum, auto

from torch import nn


class RouterType(Enum):
    """An enum for router types."""

    TOP_1 = auto()
    TOP_2 = auto()


class Router(ABC):
    pass


class Top1Router(Router, nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        pass


def get_router(router_type: RouterType) -> Router:
    pass
