# Copyright 2023 EleutherAI Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by pipegoose team


import os
import random
from typing import Dict, List, Literal

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc

from pipegoose.constants import SEED, WORKER_NAME
from pipegoose.distributed._initializers.initialize_data import (
    DataParallelGroupInitializer,
)
from pipegoose.distributed._initializers.initialize_pipeline import (
    PipelineParallelGroupInitializer,
)
from pipegoose.distributed._initializers.initialize_tensor import (
    TensorParallelGroupInitializer,
)
from pipegoose.distributed.parallel_mode import ParallelMode

DistributedBackend = Literal["gloo", "mpi", "nccl"]
RanksToDevice = Dict[ParallelMode, int]


_PARALLEL_CONTEXT = None


class ParallelContext:
    """
    Inspired from OSLO's parallel context:
    https://github.com/EleutherAI/oslo/blob/f16c73bc5893cd6cefe65e70acf6d88428a324e1/oslo/torch/distributed/parallel_context.py#L53
    """

    @classmethod
    def from_torch(
        cls,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        data_parallel_size: int,
        seed: int = SEED,
        backend: DistributedBackend = "gloo",
    ):
        """Initialize parallel context based on the environment variables defined by torchrun."""
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        host = os.environ["MASTER_ADDR"]
        port = int(os.environ["MASTER_PORT"])

        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size,
            host=host,
            port=port,
            seed=seed,
            backend=backend,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            data_parallel_size=data_parallel_size,
        )

    def __init__(
        self,
        rank: int,
        local_rank: int,
        world_size: int,
        local_world_size: int,
        host: str,
        port: int,
        seed: int,
        backend: DistributedBackend,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        data_parallel_size: int,
    ):
        """Initialize parallel context."""
        num_gpus_per_model = tensor_parallel_size * pipeline_parallel_size

        assert (
            world_size % data_parallel_size == 0
        ), "The total number of processes must be divisible by the data parallel size."
        assert world_size % num_gpus_per_model == 0, (
            "The total number of processes must be divisible by"
            "the number of GPUs per model (tensor_parallel_size * pipeline_parallel_size)."
        )
        assert num_gpus_per_model * data_parallel_size == world_size, (
            "The number of process requires to train all replicas",
            "must be equal to the world size.",
        )

        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.data_parallel_size = data_parallel_size
        self.seed = seed
        self.backend = backend

        self._global_ranks = {}
        self._local_ranks = {}
        self._world_sizes = {}
        self._groups = {}
        self._ranks_in_group = {}
        self._ranks_to_device = {}

        self.local_rank = local_rank
        self.local_world_size = local_world_size

        self.init_global_dist(rank, world_size, backend, host, port)
        self.init_parallel_groups()
        self.set_device()
        self.map_rank_to_device()

        self.rpc_worker_map = {rank: WORKER_NAME.format(rank) for rank in self.get_ranks_in_group(ParallelMode.GLOBAL)}
        self.init_rpc_workers(host, port)

        self.set_seed(seed)
        self._set_context()

    def _set_context(self):
        global _PARALLEL_CONTEXT
        _PARALLEL_CONTEXT = self

    @staticmethod
    def get_context() -> "ParallelContext":
        "Return the initialized parallel context."
        return _PARALLEL_CONTEXT

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
        dist.init_process_group(
            rank=rank,
            world_size=world_size,
            backend=backend,
            init_method=init_method,
        )
        ranks = list(range(world_size))
        process_group = dist.new_group(
            ranks=ranks,
            backend=dist.get_backend(),
        )
        self._register_dist(rank, world_size, process_group, ranks_in_group=ranks, parallel_mode=ParallelMode.GLOBAL)
        self.add_global_rank(ParallelMode.GLOBAL, rank)

    def init_parallel_groups(self):
        """Initialize 3D parallelism's all process groups."""
        rank = self.get_global_rank()
        world_size = self.get_world_size(ParallelMode.GLOBAL)

        # NOTE: ensure all processes have joined the global group
        # before creating other groups
        dist.barrier()

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

    def init_rpc_workers(self, host: str, port: int):
        """Initialize RPC workers for P2P communication."""

        if self.pipeline_parallel_size > 1:
            world_size = self.get_world_size(ParallelMode.GLOBAL)
            # NOTE: we actually only need to initialize RPC workers for
            # pipeline parallelism, but we also do so for testing all
            # the general modules that use RPC calls and work
            # in all parallel modes
            init_method = f"tcp://{host}:{port}"
            options = rpc.TensorPipeRpcBackendOptions(
                init_method=init_method,
            )

            rank = self.get_global_rank()
            worker_name = self.get_worker_name(rank)

            # NOTE: we only do device mapping for multi-gpu
            if torch.cuda.device_count() > 1:
                ranks = self.get_ranks_in_group(ParallelMode.GLOBAL)
                for other_rank in ranks:
                    if other_rank == rank:
                        continue
                    options.set_device_map(WORKER_NAME.format(other_rank), {rank: other_rank})

            rpc.init_rpc(name=worker_name, rank=rank, world_size=world_size, rpc_backend_options=options)

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

    def set_device(self):
        num_devices = torch.cuda.device_count()
        if num_devices > 0:
            device = self.get_global_rank() % num_devices
            torch.cuda.set_device(device)

    def set_seed(self, seed: int):
        """Set seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def map_rank_to_device(self):
        """Map global rank to device."""
        rank_tensor = torch.zeros(len(self._local_ranks), dtype=torch.long)

        # NOTE: if the backend is nccl, all tensors in communication must
        # be move to gpu memory
        if self.backend == "nccl":
            rank_tensor = rank_tensor.cuda()

        for idx, local_rank in enumerate(self._local_ranks.values()):
            rank_tensor[idx] = local_rank

        rank_tensor_list = [torch.zeros_like(rank_tensor) for _ in range(self.get_world_size(ParallelMode.GLOBAL))]

        dist.all_gather(tensor_list=rank_tensor_list, tensor=rank_tensor)

        for _rank, _rank_tensor in enumerate(rank_tensor_list):
            modes_and_ranks = {mode: rank for mode, rank in zip(self._local_ranks.keys(), _rank_tensor.tolist())}
            self._ranks_to_device[tuple(modes_and_ranks.items())] = _rank

    def ranks2device(self, ranks: RanksToDevice) -> int:
        """Return the global device id from ranks."""
        assert ranks in self._ranks_to_device, f"{ranks} not in {self._ranks_to_device}"
        return self._ranks_to_device[ranks]

    def is_initialized(self, parallel_mode: ParallelMode) -> bool:
        """Check if the parallel mode is initialized.

        Args:
            mode (ParallelMode): parallel mode

        Returns:
            bool: True if the parallel mode is initialized, False otherwise
        """
        return True if parallel_mode in self._groups else False

    def get_global_rank(self) -> int:
        """Get the global rank of the local process."""
        return self._global_ranks[ParallelMode.GLOBAL]

    def add_global_rank(self, parallel_mode: ParallelMode, rank: int):
        """Add the global rank of the local process."""
        self._global_ranks[parallel_mode] = rank

    def get_local_rank(self, parallel_mode: ParallelMode) -> int:
        """Get the local rank of the local process in a given parallel mode."""
        return self._local_ranks[parallel_mode]

    def add_local_rank(self, parallel_mode: ParallelMode, rank: int):
        """Add the local rank of the local process in a given parallel mode."""
        self._local_ranks[parallel_mode] = rank

    def get_global_rank_from_local_rank(self, local_rank: int, parallel_mode: ParallelMode) -> int:
        """Get the global rank from a local rank in a given parallel mode."""
        process_group = self.get_group(parallel_mode)
        return dist.get_global_rank(process_group, local_rank)

    def get_world_size(self, parallel_mode: ParallelMode) -> int:
        """Get the world size of a given parallel mode."""
        return self._world_sizes[parallel_mode]

    def add_world_size(self, parallel_mode: ParallelMode, world_size: int):
        """Add the world size of a given parallel mode."""
        self._world_sizes[parallel_mode] = world_size

    def add_group(self, parallel_mode: ParallelMode, group: dist.ProcessGroup) -> int:
        """Add a process group of a given parallel mode."""
        self._groups[parallel_mode] = group

    def get_group(self, parallel_mode: ParallelMode) -> dist.ProcessGroup:
        """Get a process group of a given parallel mode."""
        return self._groups[parallel_mode]

    def add_ranks_in_group(self, parallel_mode: ParallelMode, ranks_in_group: List[int]):
        """Add a list of global ranks in a given parallel mode of the local process."""
        self._ranks_in_group[parallel_mode] = ranks_in_group

    def get_ranks_in_group(self, parallel_mode: ParallelMode) -> List[int]:
        """A list of global ranks in a given parallel mode of the local process."""
        return self._ranks_in_group[parallel_mode]

    def get_next_global_rank(self, parallel_mode: ParallelMode) -> int:
        """Get the next global rank in a given parallel mode."""
        rank = self.get_global_rank()
        next_local_rank = self.get_next_local_rank(rank, parallel_mode)
        ranks_in_group = self.get_ranks_in_group(parallel_mode)
        next_global_rank = ranks_in_group[next_local_rank]
        return next_global_rank

    def get_prev_global_rank(self, parallel_mode: ParallelMode) -> int:
        """Get the previous global rank in a given parallel mode."""
        rank = self.get_global_rank()
        prev_local_rank = self.get_prev_local_rank(rank, parallel_mode)
        ranks_in_group = self.get_ranks_in_group(parallel_mode)
        prev_global_rank = ranks_in_group[prev_local_rank]
        return prev_global_rank

    def get_next_local_rank(self, rank, parallel_mode: ParallelMode) -> int:
        """Get the next local rank in a given parallel mode."""
        world_size = self.get_world_size(parallel_mode)
        return (rank + 1) % world_size

    def get_prev_local_rank(self, rank, parallel_mode: ParallelMode) -> int:
        """Get the previous local rank in a given parallel mode."""
        world_size = self.get_world_size(parallel_mode)
        return (rank - 1) % world_size

    def is_first_rank(self, parallel_mode: ParallelMode) -> bool:
        local_rank = self.get_local_rank(parallel_mode)
        return local_rank == 0

    def is_last_rank(self, parallel_mode: ParallelMode) -> bool:
        local_rank = self.get_local_rank(parallel_mode)
        world_size = self.get_world_size(parallel_mode)
        return local_rank == world_size - 1

    def get_worker_name(self, rank: int) -> str:
        """Return the worker name of a given rank in distributed RPC."""
        worker_name = self.rpc_worker_map[rank]
        return worker_name

    def destroy(self):
        assert self.is_initialized(ParallelMode.GLOBAL), "Global group must be initialized before destroying."
        for mode, group in self._groups.items():
            assert self.is_initialized(mode), f"{mode} group must be initialized before destroying."
            if mode is not ParallelMode.GLOBAL:
                # NOTE: only ranks in the parallel group need to synchronize
                # before destroying the group
                process_group = self.get_group(mode)
                dist.barrier(group=process_group)
                dist.destroy_process_group(group)

        dist.barrier()
        dist.destroy_process_group()

        if self.get_world_size(ParallelMode.GLOBAL) > 1:
            rpc.shutdown()

        self._groups.clear()
