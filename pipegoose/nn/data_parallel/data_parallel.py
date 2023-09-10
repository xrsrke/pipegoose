import torch
from torch import nn
import torch.distributed as dist

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode


class DataParallel:
    def __init__(self, module: nn.Module, parallel_context: ParallelContext):
        self.module = module
        self.parallel_context = parallel_context

    @torch.no_grad()
    def parallelize(self) -> nn.Module:
        module = self.module

        if self.parallel_context.data_parallel_size > 1:
            self._register_grad_avg_hook(module)

        return module

    def _register_grad_avg_hook(self, module: nn.Module):
        for p in module.parameters():
            if p.requires_grad is True:
                p.register_hook(self._average_grad)

    def _average_grad(self, grad: torch.Tensor) -> torch.Tensor:
        data_parallel_size = self.parallel_context.data_parallel_size
        process_group = self.parallel_context.get_group(ParallelMode.DATA)

        # NOTE: (grad1 + grad2 + ... + gradn) / n = grad1/n + grad2/n + ... + gradn/n
        new_grad = grad / data_parallel_size
        dist.all_reduce(new_grad, op=dist.ReduceOp.SUM, group=process_group)

        return new_grad
