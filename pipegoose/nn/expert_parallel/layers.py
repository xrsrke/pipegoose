from torch import nn
from torchtyping import TensorType

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.expert_parallel.experts import Experts
from pipegoose.nn.expert_parallel.routers import Router
from pipegoose.nn.expert_parallel.utils import get_num_local_experts
from pipegoose.nn.expert_parallel.expert_context import ExpertContext


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
        parallel_context: ParallelContext
    ):
        super().__init__()
        self.router = router
        if enable_tensor_parallel is True:
            self.num_local_experts = num_experts
        else:
            self.num_local_experts = get_num_local_experts(num_experts, parallel_context)

        self._experts = Experts(self.num_local_experts, expert, enable_tensor_parallel, parallel_context)
        self.parallel_context = parallel_context

    @property
    def experts(self) -> nn.ModuleList:
        return self._experts.experts

    def forward(self, *args, **kwargs) -> TensorType["batch_size", "seq_len", "d_model"]:
        # TODO: use torch.fx to extract the inputs from args, and kwargs
        inputs = args[0]
        router_output = self.router(inputs)
        expert_context = ExpertContext.get_instance()
        expert_context.push_aux_loss(router_output.aux_loss)
        expert_context.push_z_loss(router_output.z_loss)
        outputs = self._experts(inputs, router_output.dispatching_order, *args, **kwargs)
        return outputs
