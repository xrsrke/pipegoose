import re
from typing import Callable, List, Optional, Union

import torch
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.expert_parallel.layers import ExpertLayer
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
        expert: Optional[nn.Module] = None,
        mapping: Optional[List[int]] = None,
        router: Callable = None,
        # noise_poligy: Union[str, Callable],
        enable_tensor_parallelism: bool = False,
        parallel_context: ParallelContext = None,
    ):
        tensor_parallel_size = parallel_context.get_world_size(ParallelMode.TENSOR)
        assert parallel_context is not None, "parallel_context must be provided"
        assert num_experts % tensor_parallel_size == 0, "The number of experts must be divisible by the tensor parallel size."
        num_layers = module.config.num_hidden_layers
        assert [
            0 <= i < num_layers for i in mapping
        ], f"There is a layer index that out of range. Expected range: [0, {num_layers}-1]"

        if mapping is None:
            # NOTE: default mapping is to parallelize all MLP layers
            mapping = list(range(module.config.num_hidden_layers))

        self.module = module
        self.num_experts = num_experts
        self.expert = expert
        self.mapping = mapping
        self.router = router
        # self.noise_policy = noise_poligy
        self.enable_tensor_parallelism = enable_tensor_parallelism
        self.parallel_context = parallel_context

    @torch.no_grad()
    def parallelize(self) -> nn.Module:
        # TODO: make it generalize
        def _is_mlp(name) -> Union[bool, Optional[int]]:
            pattern = re.compile(r"^transformer\.h\.(\d+)\.mlp$")
            match = pattern.match(name)
            if match:
                layer_idx = int(match.group(1))
                return True, layer_idx
            else:
                return False, None

        for name, module in self.module.named_modules():
            is_mlp, layer_idx = _is_mlp(name)
            if is_mlp:
                if layer_idx in self.mapping:
                    expert_layer = ExpertLayer(
                        self.num_experts,
                        module if self.expert is None else self.expert,
                        self.router,
                        self.enable_tensor_parallelism,
                        self.parallel_context,
                    )
                    # TODO: make it generalize
                    getattr(self.module, "transformer").h[layer_idx].mlp = expert_layer

        return self.module

    @torch.no_grad()
    def deparallelize(self):
        pass
