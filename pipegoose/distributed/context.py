# import os
from enum import Enum
from typing import List, Literal

import torch.distributed as dist

from pipegoose.distributed._initializers.initialize_data import (
    DataParallelGroupInitializer,
)
from pipegoose.distributed._initializers.initialize_pipeline import (
    PipelineParallelGroupInitializer,
)
from pipegoose.distributed._initializers.initialize_tensor import (
    TensorParallelGroupInitializer,
)


class ParallelMode(Enum):
    GLOBAL = "global"

    TENSOR = "tensor"
    PIPELINE = "pipeline"
    DATA = "data"


DistributedBackend = Literal["gloo", "mpi", "nccl"]


class ParallelContext:
    """Inspired from OSLO's parallel context:
    https://github.com/EleutherAI/oslo/blob/f16c73bc5893cd6cefe65e70acf6d88428a324e1/oslo/torch/distributed/parallel_context.py#L53
    """

    def __init__(
        self,
        rank: int,
        local_rank: int,
        world_size: int,
        local_world_size: int,
        host: str,
        port: int,
        backend: DistributedBackend,
        seed: int,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        data_parallel_size: int,
    ):
        assert world_size == tensor_parallel_size * pipeline_parallel_size, (
            "The total number of processes must be equal to the product of the ",
            "tensor parallel size and the pipeline parallel size.",
        )

        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.data_parallel_size = data_parallel_size

        self._global_ranks = {}
        self._local_ranks = {}
        self._world_sizes = {}
        self._groups = {}
        self._ranks_in_group = {}
        self._rank_to_devices = {}

        self.init_global_dist(rank, world_size, backend, host, port)
        self.init_parallel_groups()

    def init_global_dist(self, rank: int, world_size: int, backend: DistributedBackend, host: str, port: int):
        """Initialize the global distributed group.

        Args:
            rank (int): global rank
            world_size (int): global world size
            backend (DistributedBackend): distributed backend
            host (str): communication host
            port (int): communication port
        """
        process_group = dist.init_process_group(rank=rank, world_size=world_size, backend=backend, host=host, port=port)
        ranks = list(range(world_size))
        self._register_dist(rank, world_size, process_group, ranks_in_group=ranks, mode=ParallelMode.GLOBAL)
        self.add_global_rank(ParallelMode.GLOBAL, rank)

    def init_parallel_groups(self):
        rank = self.get_global_rank()
        world_size = self.get_world_size(ParallelMode.GLOBAL)

        initializer_params = {
            "rank": rank,
            "world_size": world_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "data_parallel_size": self.data_parallel_size,
        }

        initializer_results = [
            TensorParallelGroupInitializer(**initializer_params).init_dist_group(),
            PipelineParallelGroupInitializer(**initializer_params).init_dist_group(),
            DataParallelGroupInitializer(**initializer_params).init_dist_group(),
        ]

        for initializer_result in initializer_results:
            self._register_dist(**initializer_result)

    def _register_dist(
        self,
        local_rank: int,
        local_world_size: int,
        process_group: dist.ProcessGroup,
        ranks_in_group: List[int],
        mode: ParallelMode,
    ):
        """Register distributed group based on the parallel mode.

        Args:
            local_rank (int): local rank
            local_world_size (int): local world size
            mode (ParallelMode): parallel mode
        """
        self.add_local_rank(mode, local_rank)
        self.add_world_size(mode, local_world_size)
        self.add_group(mode, process_group)
        self.add_ranks_in_group(mode, ranks_in_group)

    def get_global_rank(self) -> int:
        return self.get_global_rank[ParallelMode.GLOBAL]

    def add_global_rank(self, mode: ParallelMode, rank: int) -> int:
        self._global_ranks[mode] = rank

    def get_local_rank(self, mode: ParallelMode) -> int:
        return self._local_ranks[mode]

    def add_local_rank(self, mode: ParallelMode, rank: int) -> int:
        self._local_ranks[mode] = rank

    def get_world_size(self, mode: ParallelMode) -> int:
        return self._world_size[mode]

    def add_world_size(self, mode: ParallelMode, world_size: int) -> int:
        self._local_world_size[mode] = world_size

    def add_group(self, mode: ParallelMode, group: dist.ProcessGroup) -> int:
        self._groups[mode] = group

    def add_ranks_in_group(self, mode: ParallelMode, ranks_in_group: List[int]) -> int:
        self._ranks_in_group[mode] = ranks_in_group
