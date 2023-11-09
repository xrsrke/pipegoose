import torch
from torch import nn
from torchtyping import TensorType

from pipegoose.distributed.functional import all_gather
from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode


class Experts(nn.Module):
    """A collection of experts in an expert layer."""

    def __init__(
        self,
        num_experts: int,
        expert: nn.Module,
        enable_tensor_parallel: bool,
        parallel_context: ParallelContext,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.enable_tensor_parallel = enable_tensor_parallel
        self.parallel_context = parallel_context

        if enable_tensor_parallel is True:
            num_local_experts = num_experts
        else:
            expert_parallel_size = parallel_context.get_world_size(ParallelMode.TENSOR)
            num_local_experts = num_experts // expert_parallel_size

        expert = expert() if not isinstance(expert, nn.Module) else expert
        self.experts = nn.ModuleList([expert for _ in range(num_local_experts)])

    def forward(
        self,
        inputs: TensorType["batch_size", "seq_len", "d_model"],
        dispatch_order: TensorType["batch_size * seq_len"],
    ) -> TensorType["batch_size", "seq_len", "d_model"]:
        outputs = []
        for expert in self.experts:
            embeddings = self._get_dispatch_embeddings(inputs, dispatch_order, expert_idx=1)
            outputs.append(expert(embeddings))

        outputs = torch.cat([outputs], dim=0)
        all_gather(
            outputs,
            dim=0,
            parallel_context=self.parallel_context,
            parallel_mode=ParallelMode.TENSOR,
        )
        return outputs

    def _get_dispatch_embeddings(embeddings: torch.Tensor, dispatch_order: torch.TensorType, expert_idx: int) -> torch.Tensor:
        """Dispatch embeddings to the corresponding expert."""
        token_indices = [i for i, e in enumerate(dispatch_order) if e == expert_idx]
        dispatched_embeddings = embeddings[token_indices]
        return dispatched_embeddings
