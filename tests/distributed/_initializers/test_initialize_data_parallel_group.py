import pytest
import torch
import torch.distributed as dist
from utils import map_rank_to_group

from pipegoose.distributed._initializers.initialize_data import (
    DataParallelGroupInitializer,
)
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.testing.utils import spawn

GROUPS_IN_WORLD_SIZE_1 = [0]
GROUPS_IN_WORLD_SIZE_8 = [[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11], [12, 14], [13, 15]]


def init_tensor_parallel_group(
    rank, world_size, host, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, groups
):
    init_method = f"tcp://{host}:{port}"
    expected_group = map_rank_to_group(rank, groups)

    torch.distributed.init_process_group(
        rank=rank,
        world_size=world_size,
        backend="gloo",
        init_method=init_method,
    )

    result = DataParallelGroupInitializer(
        rank,
        world_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    ).init_dist_group()

    assert isinstance(result["local_rank"], int)
    assert isinstance(result["local_world_size"], int)
    # TODO: how to assert process_group?
    assert result["process_group"] is not None
    assert isinstance(result["ranks_in_group"], list)
    assert result["ranks_in_group"] == expected_group
    assert result["parallel_mode"] == ParallelMode.DATA

    dist.barrier()
    dist.destroy_process_group(result["process_group"])
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, groups",
    [(1, 1, 1, 1, GROUPS_IN_WORLD_SIZE_1), (8, 2, 2, 2, GROUPS_IN_WORLD_SIZE_8)],
)
def test_init_tensor_parallel_group(world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, groups):
    spawn(
        init_tensor_parallel_group,
        nprocs=world_size,
        host="localhost",
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        groups=groups,
    )
