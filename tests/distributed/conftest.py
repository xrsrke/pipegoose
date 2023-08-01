import pytest

from pipegoose.distributed.parallel_context import ParallelContext


@pytest.fixture(scope="session")
def parallel_context():
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1
    SEED = 69
    RANK = 0
    WORLD_SIZE = 1
    HOST = "localhost"
    PORT = 12355

    parallel_context = ParallelContext(
        rank=RANK,
        local_rank=RANK,
        world_size=WORLD_SIZE,
        local_world_size=WORLD_SIZE,
        host=HOST,
        port=PORT,
        backend="gloo",
        seed=SEED,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
    )

    return parallel_context
