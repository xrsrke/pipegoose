"""DON'T USE THIS MODULE: Under development."""

from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext


class Experts(nn.Module):
    def __init__(self, num_experts: int, expert: nn.Module, parallel_context: ParallelContext):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([expert for _ in range(num_experts)])
        self.parallel_context = parallel_context

    def forward(self):
        pass
