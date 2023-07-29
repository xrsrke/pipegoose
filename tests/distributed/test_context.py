import pytest
from torch.multiprocessing import Process

from pipegoose.distributed.context import ParallelContext
from pipegoose.distributed.mode import ParallelMode


def run_worker(rank, world_size):
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    parallel_context = ParallelContext(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        host="localhost",
        port=12355,
        backend="gloo",
        seed=69,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=1,
    )

    try:
        # assert parallel_context.rank == rank
        parallel_modes = [
            # ParallelMode.TENSOR,
            # ParallelMode.PIPELINE,
            ParallelMode.DATA,
        ]

        assert parallel_context.is_initialized() is True
        assert parallel_context.tensor_parallel_size == TENSOR_PARALLEL_SIZE
        assert parallel_context.pipeline_parallel_size == PIPELINE_PARALLEL_SIZE
        assert parallel_context.data_parallel_size == DATA_PARALLEL_SIZE

        assert parallel_context.get_global_rank() == rank

        # for parallel_mode in parallel_modes:
        #     assert type(parallel_context.get_local_rank(parallel_mode)) == int
        #     assert type(parallel_context.get_world_size(parallel_mode)) == int
    except Exception as e:
        pytest.fail(f"assertion failed: {e}")


def test_parallel_context():
    world_size = 4
    processes = []

    for rank in range(world_size):
        p = Process(target=run_worker, args=(rank, world_size))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        assert p.exitcode == 0
