import pytest
import torch.distributed as dist

from pipegoose.distributed._initializers.initialize_sequence import (
    SequenceParallelGroupInitializer,
)
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.testing.utils import spawn

GROUPS_IN_WORLD_SIZE_1 = [0]
GROUPS_IN_WORLD_SIZE_8 = [[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11], [12, 14], [13, 15]]


def init_tensor_parallel_group(
    rank,
    world_size,
    host,
    port,
    tensor_parallel_size,
    pipeline_parallel_size,
    data_parallel_size,
    sequence_parallel_size,
    groups,
):
    init_method = f"tcp://{host}:{port}"
    # expected_ranks = map_rank_to_group(rank, groups)

    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend="gloo",
        init_method=init_method,
    )

    result = SequenceParallelGroupInitializer(
        rank,
        world_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        sequence_parallel_size=sequence_parallel_size,
    ).init_dist_group()

    assert 1 == 1

    assert 0 <= result["local_rank"] < result["local_world_size"]
    assert result["local_rank"] < sequence_parallel_size

    assert result["local_world_size"] == sequence_parallel_size

    assert isinstance(result["process_group"], dist.ProcessGroup)

    # assert result["ranks_in_group"] == expected_ranks
    # assert dist.get_process_group_ranks(result["process_group"]) == expected_ranks
    assert result["parallel_mode"] == ParallelMode.SEQUENCE

    dist.barrier()
    dist.destroy_process_group(result["process_group"])
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "sequence_parallel_size, groups",
    [(1, GROUPS_IN_WORLD_SIZE_1), (2, GROUPS_IN_WORLD_SIZE_8)],
)
def test_init_tensor_parallel_group(sequence_parallel_size, groups):
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    WORLD_SIZE = TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE * DATA_PARALLEL_SIZE * sequence_parallel_size

    spawn(
        init_tensor_parallel_group,
        world_size=WORLD_SIZE,
        host="localhost",
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        sequence_parallel_size=sequence_parallel_size,
        groups=groups,
    )
