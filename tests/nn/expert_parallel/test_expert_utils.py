import pytest

from pipegoose.nn.expert_parallel.utils import get_num_local_experts
from pipegoose.testing.utils import init_parallel_context, spawn


def run_get_num_local_experts(
    rank,
    world_size,
    port,
    tensor_parallel_size,
    pipeline_parallel_size,
    data_parallel_size,
    num_experts,
    ref_num_local_experts,
):
    parallel_context = init_parallel_context(
        rank,
        world_size,
        port,
        tensor_parallel_size,
        pipeline_parallel_size,
        data_parallel_size,
    )

    num_local_experts = get_num_local_experts(num_experts, parallel_context)

    assert num_local_experts == ref_num_local_experts


@pytest.mark.parametrize(
    "tensor_parallel_size, num_experts, expected",
    [
        (1, 16, 16),
        (2, 16, 8),
        (4, 16, 4),
        (8, 16, 2),
    ],
)
def test_get_num_local_experts(tensor_parallel_size, num_experts, expected):
    DATA_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 1
    WORLD_SIZE = tensor_parallel_size * PIPELINE_PARALLEL_SIZE * DATA_PARALLEL_SIZE

    spawn(
        run_get_num_local_experts,
        world_size=WORLD_SIZE,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        num_experts=num_experts,
        ref_num_local_experts=expected,
    )
