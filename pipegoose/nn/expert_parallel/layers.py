from torch import nn
from torchtyping import TensorType

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.expert_parallel.experts import Experts
from pipegoose.nn.expert_parallel.routers import Router


class ExpertLayer(nn.Module):
    """
    An expert layer.

    NOTE: Switch Transformer: https://arxiv.org/abs/2101.03961
    """

    def __init__(
        self,
        num_experts: int,
        expert: nn.Module,
        router: Router,
        enable_tensor_parallel: bool,
        parallel_context: ParallelContext,
    ):
        super().__init__()
        self.router = router
        self._experts = Experts(num_experts, expert, enable_tensor_parallel, parallel_context)
        self.parallel_context = parallel_context

    @property
    def experts(self) -> nn.ModuleList:
        return self._experts.experts

    def forward(
        self, inputs: TensorType["batch_size", "seq_len", "d_model"]
    ) -> TensorType["batch_size", "seq_len", "d_model"]:
        dispatching_order, _, _ = self.router(inputs)
        outputs = self._experts(inputs, dispatching_order)
        return outputs
