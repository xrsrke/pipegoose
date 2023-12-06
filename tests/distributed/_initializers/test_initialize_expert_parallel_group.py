import pytest
import torch.distributed as dist
from utils import map_rank_to_group

from pipegoose.distributed._initializers.initialize_expert import (
    ExpertDataParallelGroupInitializer,
)
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.testing.utils import spawn

GROUPS_IN_WORLD_SIZE_1 = [0]
GROUPS_IN_WORLD_SIZE_8 = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]


def init_tensor_parallel_group(
    rank, world_size, host, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, groups
):
    init_method = f"tcp://{host}:{port}"
    expected_ranks = map_rank_to_group(rank, groups)

    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend="gloo",
        init_method=init_method,
    )

    result = ExpertDataParallelGroupInitializer(
        rank,
        world_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    ).init_dist_group()

    assert 0 <= result["local_rank"] < result["local_world_size"]
    assert result["local_rank"] < tensor_parallel_size

    assert result["local_world_size"] == tensor_parallel_size

    assert isinstance(result["process_group"], dist.ProcessGroup)

    assert result["ranks_in_group"] == expected_ranks
    assert dist.get_process_group_ranks(result["process_group"]) == expected_ranks

    assert result["parallel_mode"] == ParallelMode.EXPERT_DATA

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
        world_size=world_size,
        host="localhost",
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        groups=groups,
    )
