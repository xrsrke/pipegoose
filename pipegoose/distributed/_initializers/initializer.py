from abc import ABC, abstractclassmethod
from typing import TypedDict

from torch.distributed import ProcessGroup

from pipegoose.distributed.mode import ParallelMode


class ProcessGroupResult(TypedDict):
    local_rank: int
    local_world_size: int
    process_group: ProcessGroup
    parallel_mode: ParallelMode


class ProcessGroupInitializer(ABC):
    def __init__(
        self, rank: int, world_size: int, tensor_parallel_size: int, pipeline_parallel_size: int, data_parallel_size: int
    ):
        self.rank = rank
        self.world_size = world_size
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.data_parallel_size = data_parallel_size

    @abstractclassmethod
    def init_dist_group(self) -> ProcessGroupResult:
        raise NotImplementedError
