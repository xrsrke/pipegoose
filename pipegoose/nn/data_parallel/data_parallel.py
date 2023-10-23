import torch
import torch.distributed as dist
from torch import nn

from pipegoose.distributed.functional import all_reduce
from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.parallel import Parallel


class DataParallel(Parallel):
    def __init__(self, module: nn.Module, parallel_context: ParallelContext):
        self.module = module
        self.parallel_context = parallel_context

    @torch.no_grad()
    def parallelize(self) -> nn.Module:
        module = self.module

        if self.parallel_context.data_parallel_size > 1:
            self._register_grad_avg_hook(module)
            self._save_metadata(module, self.parallel_context)

        return module

    def _register_grad_avg_hook(self, module: nn.Module):
        for p in module.parameters():
            if p.requires_grad is True:
                p.register_hook(self._average_grad)

    def _average_grad(self, grad: torch.Tensor):
        # NOTE: (grad1 + grad2 + ... + gradn) / n = grad1/n + grad2/n + ... + gradn/n
        grad.div_(self.parallel_context.data_parallel_size)
        all_reduce(grad, op=dist.ReduceOp.SUM, parallel_context=self.parallel_context, parallel_mode=ParallelMode.DATA)
