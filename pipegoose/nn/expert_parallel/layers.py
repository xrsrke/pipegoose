from torch import nn
from torchtyping import TensorType

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.expert_parallel.experts import Experts
from pipegoose.nn.expert_parallel.routers import Router


class ExpertLayer(nn.Module):
    """
    An expert layer.

    NOTE:
    Switch Transformer: https://arxiv.org/abs/2101.03961
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
        self.experts = Experts(num_experts, expert, enable_tensor_parallel, parallel_context)

    def forward(
        self, inputs: TensorType["batch_size", "seq_len", "d_model"]
    ) -> TensorType["batch_size", "seq_len", "d_model"]:
        dispatching_order, _, _ = self.router(inputs, self.experts)
        outputs = self.experts(inputs, dispatching_order)
        return outputs
