import torch
from torch import nn


class ParamHook:
    def __init__(self, module: nn.Module):
        self.module = module

    def register_pre_backward_hook(self):
        if not torch.is_grad_enabled():
            return

        for p in self.module.parameters():
            if p.requires_grad:
                p_temp = p.expand_as(p)
                assert p_temp.grad_fn is not None

    def register_post_backward_hook(self):
        pass
