import pytest

from pipegoose.nn.pipeline_parallel2._utils import get_partition_idx
from pipegoose.testing.utils import init_parallel_context, spawn

# NOTE: a mapping from global rank to partition index in pipeline parallelism
# (tensor_parallel_size, pipeline_parallel_size, data_parallel_size) = {rank: partition_idx}
RANK_TO_PARTITION_IDX = {
    (2, 4, 2): {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 1,
        5: 1,
        6: 1,
        7: 1,
        8: 2,
        9: 2,
        10: 2,
        11: 2,
        12: 3,
        13: 3,
        14: 3,
        15: 3,
    },
    (1, 4, 1): {0: 0, 1: 1, 2: 2, 3: 3},
}


def run_get_partition_idx(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, rank_to_partition_idx
):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    partition_idx = get_partition_idx(parallel_context)

    assert partition_idx == rank_to_partition_idx[rank]


@pytest.mark.parametrize("tensor_parallel_size, pipeline_parallel_size, data_parallel_size", [(2, 4, 2), (1, 4, 1)])
def test_get_partition_idx(tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    world_size = tensor_parallel_size * pipeline_parallel_size * data_parallel_size

    rank_to_partition_idx = RANK_TO_PARTITION_IDX[(tensor_parallel_size, pipeline_parallel_size, data_parallel_size)]

    spawn(
        run_get_partition_idx,
        world_size=world_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        rank_to_partition_idx=rank_to_partition_idx,
    )
