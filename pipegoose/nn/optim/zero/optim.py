from typing import Any, Dict, Iterable, List

import torch
from torch import nn
from torch.optim import Optimizer

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.optim import DistributedOptimizer
from pipegoose.nn.optim.zero.sharding import ParameterSharding


class ZeroRedundancyOptimizer(DistributedOptimizer):
    def __init__(
        self,
        params: Iterable[List[nn.Parameter]],
        optim: Optimizer,
        parallel_context: ParallelContext,
        sharding: ParameterSharding,
        **default: Any
    ):
        # TODO: accept params, and optim separately
        self.params = params
        self._optim_constructor = optim
        self.parallel_context = parallel_context
        self.default = default

        # world_size = parallel_context.get_world_size(ParallelMode.GLOBAL)

    def _partition_and_move_to_rank(self):
        self._partition_params(self.optim.param_groups)

    def _construct_local_optim(self, local_params: Dict[str, torch.Tensor]):
        self.optim = self._optim_constructor(local_params, **self.default)

    def _sync_hyperparams(self, source: List[Dict[Any, Any]], destination: List[Dict[Any, Any]]):
        for source_group, destination_group in zip(source, destination):
            for k in source_group.keys():
                if k != "params":
                    destination_group[k] = source_group[k]

    def setup(self):
        pass

    def step(self, **kwargs):
        self._sync_hyperparams(self.param_groups, self.optim.param_groups)

        self.optim.step(**kwargs)
