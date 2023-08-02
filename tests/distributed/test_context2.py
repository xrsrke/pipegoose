import pytest

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.testing.utils import spawn


def init_parallel_context(seed, backend, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    parallel_context = ParallelContext.from_torch(
        seed=seed,
        backend=backend,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )

    # assert parallel_context.get_world_size(ParallelMode.GLOBAL)


@pytest.mark.parametrize(
    "world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size", [(1, 1, 1, 1), (8, 2, 2, 2)]
)
def test_init_parallel_context(world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    SEED = 69
    spawn(
        init_parallel_context,
        nprocs=world_size,
        seed=SEED,
        backend="gloo",
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )
