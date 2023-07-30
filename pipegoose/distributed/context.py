import random
from typing import List, Literal

import torch
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
from pipegoose.distributed.mode import ParallelMode

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
        num_gpus_per_model = tensor_parallel_size * pipeline_parallel_size

        assert world_size == tensor_parallel_size * pipeline_parallel_size, (
            "The total number of processes must be equal to the product of the ",
            "tensor parallel size and the pipeline parallel size.",
        )
        assert (
            world_size % data_parallel_size == 0
        ), "The total number of processes must be divisible by the data parallel size."
        assert world_size == num_gpus_per_model * data_parallel_size, (
            "The total number of processes must be equal to the product of the ",
            "number of GPUs per model and the data parallel size.",
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

        self.local_rank = local_rank
        self.local_world_size = local_world_size

        self.init_global_dist(rank, world_size, backend, host, port)
        self.init_parallel_groups()
        # self.set_seed(seed)

    def init_global_dist(self, rank: int, world_size: int, backend: DistributedBackend, host: str, port: int):
        """Initialize the global distributed group.

        Args:
            rank (int): global rank
            world_size (int): global world size
            backend (DistributedBackend): distributed backend
            host (str): communication host
            port (int): communication port
        """
        init_method = f"tcp://{host}:{port}"
        process_group = dist.init_process_group(
            rank=rank,
            world_size=world_size,
            backend=backend,
            init_method=init_method,
        )
        ranks = list(range(world_size))
        self._register_dist(rank, world_size, process_group, ranks_in_group=ranks, parallel_mode=ParallelMode.GLOBAL)
        self.add_global_rank(ParallelMode.GLOBAL, rank)

    def init_parallel_groups(self):
        rank = self.get_global_rank()
        world_size = self.get_world_size(ParallelMode.GLOBAL)

        params = {
            "rank": rank,
            "world_size": world_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "data_parallel_size": self.data_parallel_size,
        }

        results = [
            TensorParallelGroupInitializer(**params).init_dist_group(),
            PipelineParallelGroupInitializer(**params).init_dist_group(),
            DataParallelGroupInitializer(**params).init_dist_group(),
        ]

        for result in results:
            self._register_dist(**result)

    def _register_dist(
        self,
        local_rank: int,
        local_world_size: int,
        process_group: dist.ProcessGroup,
        ranks_in_group: List[int],
        parallel_mode: ParallelMode,
    ):
        """Register distributed group based on the parallel mode.

        Args:
            local_rank (int): local rank
            local_world_size (int): local world size
            mode (ParallelMode): parallel mode
        """
        self.add_local_rank(parallel_mode, local_rank)
        self.add_world_size(parallel_mode, local_world_size)
        self.add_group(parallel_mode, process_group)
        self.add_ranks_in_group(parallel_mode, ranks_in_group)

    def set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)

        # TODO: set GPU seed
        if torch.cuda.is_available():
            # parallel_seed = seed
            pass

    def is_initialized(self, parallel_mode: ParallelMode) -> bool:
        """Check if the parallel mode is initialized.

        Args:
            mode (ParallelMode): parallel mode

        Returns:
            bool: True if the parallel mode is initialized, False otherwise
        """
        return True if parallel_mode in self._groups else False

    def get_global_rank(self) -> int:
        return self._global_ranks[ParallelMode.GLOBAL]

    def add_global_rank(self, parallel_mode: ParallelMode, rank: int) -> int:
        self._global_ranks[parallel_mode] = rank

    def get_local_rank(self, parallel_mode: ParallelMode) -> int:
        return self._local_ranks[parallel_mode]

    def add_local_rank(self, parallel_mode: ParallelMode, rank: int) -> int:
        self._local_ranks[parallel_mode] = rank

    def get_world_size(self, parallel_mode: ParallelMode) -> int:
        return self._world_sizes[parallel_mode]

    def add_world_size(self, parallel_mode: ParallelMode, world_size: int) -> int:
        self._world_sizes[parallel_mode] = world_size

    def add_group(self, parallel_mode: ParallelMode, group: dist.ProcessGroup) -> int:
        self._groups[parallel_mode] = group

    def get_group(self, parallel_mode: ParallelMode) -> dist.ProcessGroup:
        return self._groups[parallel_mode]

    def add_ranks_in_group(self, parallel_mode: ParallelMode, ranks_in_group: List[int]) -> int:
        self._ranks_in_group[parallel_mode] = ranks_in_group

    def get_ranks_in_group(self, parallel_mode: ParallelMode) -> List[int]:
        return self._ranks_in_group[parallel_mode]
