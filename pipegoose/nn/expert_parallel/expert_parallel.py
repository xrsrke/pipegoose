from typing import Callable

import torch
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.parallel import Parallel


class ExpertParallel(Parallel):
    """
    Turn a model into an Mixture of Experts model.

    NOTE: The architecture is based on "Pipeline MoE: A Flexible MoE Implementation with Pipeline Parallelism" by Xin Chen et al.
    https://arxiv.org/abs/2304.11414
    """

    def __init__(
        self,
        module: nn.Module,
        num_experts: int,
        expert: nn.Module,
        router: Callable,
        noise_poligy: Callable,
        enable_tensor_parallelism: bool = True,
        parallel_context: ParallelContext = None,
    ):
        assert parallel_context is not None, "parallel_context must be provided"
        self.module = module
        self.num_experts = num_experts
        self.expert = expert
        self.router = router
        self.noise_policy = noise_poligy
        self.enable_tensor_parallelism = enable_tensor_parallelism
        self.parallel_context = parallel_context

    @torch.no_grad()
    def parallelize(self):
        expert_parallel_size = self.parallel_context.get_world_size(ParallelMode.TENSOR)
        assert (
            self.num_experts % expert_parallel_size == 0
        ), "The number of experts must be divisible by the tensor parallel size."

    @torch.no_grad()
    def deparallelize(self):
        pass
