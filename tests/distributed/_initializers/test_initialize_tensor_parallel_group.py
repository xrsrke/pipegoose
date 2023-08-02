import pytest
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from pipegoose.distributed._initializers.initialize_tensor import (
    TensorParallelGroupInitializer,
)
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.testing.utils import spawn


def init_tensor_parallel_group(rank, world_size, host, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    init_method = f"tcp://{host}:{port}"

    torch.distributed.init_process_group(
        rank=rank,
        world_size=world_size,
        backend="gloo",
        init_method=init_method,
    )

    result = TensorParallelGroupInitializer(
        rank,
        world_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    ).init_dist_group()

    assert isinstance(result["local_rank"], int)
    assert isinstance(result["local_world_size"], int)
    assert isinstance(result["process_group"], ProcessGroup)
    assert isinstance(result["ranks_in_group"], list)
    assert result["parallel_mode"] == ParallelMode.TENSOR

    dist.destroy_process_group(result["process_group"])
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size",
    [(1, 1, 1, 1), (8, 2, 2, 2)],
)
def test_init_tensor_parallel_group(world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    port = 14511
    spawn(
        init_tensor_parallel_group,
        nprocs=world_size,
        host="localhost",
        port=port,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )
