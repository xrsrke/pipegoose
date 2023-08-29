from abc import ABC, abstractclassmethod
from typing import List

import torch
from torch import nn
from transformers import AutoModel


class ProfileStrategy(ABC):
    def __init__(self, module: AutoModel, device: torch.device):
        self.module = module
        self.device = device

    @abstractclassmethod
    def profile(self):
        raise NotImplementedError("Not implemented.")


class ProfileByMemory(ProfileStrategy):
    """Profiles CUDA memory usage by layer."""

    def profile(self, input: torch.Tensor) -> List[int]:
        sizes = []
        input = input.to(self.device)
        output = input

        for _, layer in self.module.named_children():
            layer.to(self.device)
            layer.train()

            # calculate the memory occupied by the layer's output
            memory_before = torch.cuda.memory_allocated(device=self.device)
            output = layer(output)
            memory_after = torch.cuda.memory_allocated(device=self.device)
            occupied_memory = memory_after - memory_before

            # now calculate the memory occupied by the layer's parameters
            param_memory = self._compute_param_memory(layer)
            total_memory = occupied_memory + param_memory

            sizes.append(total_memory)
        return sizes

    def _compute_param_memory(self, module: nn.Module) -> int:
        total_size = 0
        for p in module.parameters():
            total_size += p.storage().size() * p.storage().element_size()

        return total_size
