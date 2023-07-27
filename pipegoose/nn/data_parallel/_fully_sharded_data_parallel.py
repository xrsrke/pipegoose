from typing import Any, Dict

import torch
from torch import nn

from pipegoose.distributed.context import ParallelContext
from pipegoose.nn.data_parallel.sharding import GreedySharding, ShardingStategy


class FullyShardedDataParallel(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        sharding: ShardingStategy = GreedySharding(),
        parallel_context: ParallelContext = None,
    ):
        super().__init__()

        self.module = sharding.shard(module, parallel_context)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self._rebuild_full_params()
        output = self.module(**args, **kwargs)

        return output

    def _lazy_init(self):
        self._streams: Dict[str, torch.cuda.Stream] = {}

    def pre_forward_hook(self):
        pass

    def post_forward_hook(self):
        pass

    def _rebuild_full_params(self):
        pass

    def _register_backward_hook(self):
        pass
