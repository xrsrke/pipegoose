import pytest
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.testing.utils import spawn

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

backend = ["gloo", pytest.param("nccl", marks=skip_if_no_cuda)]


# NOTE: map local rank to next rank based in [world_size][parallel_mode][local_rank]
# for world_size = 8
LOCAL_RANK_TO_NEXT_RANK = {
    1: {
        ParallelMode.TENSOR: {0: 0},
        ParallelMode.EXPERT_DATA: {0: 0},
        ParallelMode.PIPELINE: {0: 0},
        ParallelMode.DATA: {0: 0},
        ParallelMode.GLOBAL: {0: 0},
    },
    8: {
        ParallelMode.TENSOR: {0: 1, 1: 0},
        ParallelMode.EXPERT_DATA: {0: 1, 1: 0},
        ParallelMode.PIPELINE: {
            0: 1,
            1: 0,
        },
        ParallelMode.DATA: {0: 1, 1: 0},
        ParallelMode.GLOBAL: {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 0},
    },
}

LOCAL_RANK_TO_PREV_RANK = {
    1: {
        ParallelMode.TENSOR: {0: 0},
        ParallelMode.EXPERT_DATA: {0: 0},
        ParallelMode.PIPELINE: {0: 0},
        ParallelMode.DATA: {0: 0},
        ParallelMode.GLOBAL: {0: 0},
    },
    8: {
        ParallelMode.TENSOR: {0: 1, 1: 0},
        ParallelMode.EXPERT_DATA: {0: 1, 1: 0},
        ParallelMode.PIPELINE: {
            0: 1,
            1: 0,
        },
        ParallelMode.DATA: {0: 1, 1: 0},
        ParallelMode.GLOBAL: {0: 7, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6},
    },
}


def init_parallel_context(
    rank, world_size, seed, backend, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
):
    parallel_modes = [
        ParallelMode.GLOBAL,
        ParallelMode.TENSOR,
        ParallelMode.PIPELINE,
        ParallelMode.DATA,
    ]

    parallel_context = ParallelContext(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        host="localhost",
        port=port,
        seed=seed,
        backend=backend,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )

    assert parallel_context.tensor_parallel_size == tensor_parallel_size
    assert parallel_context.pipeline_parallel_size == pipeline_parallel_size
    assert parallel_context.data_parallel_size == data_parallel_size
    assert ParallelContext.get_context() == parallel_context

    assert parallel_context.get_global_rank() == rank

    for parallel_mode in parallel_modes:
        local_rank = parallel_context.get_local_rank(parallel_mode)

        assert parallel_context.is_initialized(parallel_mode) is True
        assert isinstance(parallel_context.get_group(parallel_mode), ProcessGroup)

        assert type(parallel_context.get_local_rank(parallel_mode)) == int
        assert type(parallel_context.get_world_size(parallel_mode)) == int
        assert isinstance(parallel_context.get_global_rank_from_local_rank(local_rank, parallel_mode), int)

        process_group = parallel_context.get_group(parallel_mode)
        ranks_in_group = parallel_context.get_ranks_in_group(parallel_mode)
        assert isinstance(ranks_in_group, list)
        assert ranks_in_group == dist.get_process_group_ranks(process_group)
        assert len(ranks_in_group) == parallel_context.get_world_size(parallel_mode)

        # TODO: fix this, the LOCAL_RANK_TO_NEXT_RANK is only valid for world_size = 1 and world_size = 8
        if world_size == 8 or world_size == 1:
            next_local_rank = parallel_context.get_next_local_rank(local_rank, parallel_mode)
            assert next_local_rank == LOCAL_RANK_TO_NEXT_RANK[world_size][parallel_mode][local_rank]

            prev_local_rank = parallel_context.get_prev_local_rank(local_rank, parallel_mode)
            assert prev_local_rank == LOCAL_RANK_TO_PREV_RANK[world_size][parallel_mode][local_rank]

        # TODO: fix tensor_parallel_size = 1, pipeline_parallel_size = 2, data_parallel_size = 2
        # get_ranks_in_group only return 1
        next_global_rank = parallel_context.get_next_global_rank(parallel_mode)
        assert isinstance(next_global_rank, int)

        prev_global_rank = parallel_context.get_prev_global_rank(parallel_mode)
        assert isinstance(prev_global_rank, int)

        assert parallel_context.is_first_rank(parallel_mode) == (local_rank == 0)
        assert parallel_context.is_last_rank(parallel_mode) == (
            local_rank == parallel_context.get_world_size(parallel_mode) - 1
        )

    parallel_context.destroy()

    for parallel_mode in parallel_modes:
        assert parallel_context.is_initialized(parallel_mode) is False


@pytest.mark.parametrize("tensor_parallel_size", (2,))
@pytest.mark.parametrize("pipeline_parallel_size", (4,))
@pytest.mark.parametrize("data_parallel_size", (2,))
def test_init_parallel_context(tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    SEED = 69
    BACKEND = "gloo"

    WORLD_SIZE = tensor_parallel_size * pipeline_parallel_size * data_parallel_size

    spawn(
        init_parallel_context,
        world_size=WORLD_SIZE,
        seed=SEED,
        backend=BACKEND,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )


def run_device_mapping_in_parallel_context(
    rank, world_size, seed, backend, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
):
    parallel_context = ParallelContext(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        host="localhost",
        port=port,
        seed=seed,
        backend=backend,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )

    ranks = (
        (ParallelMode.GLOBAL, parallel_context.get_local_rank(ParallelMode.GLOBAL)),
        (ParallelMode.TENSOR, parallel_context.get_local_rank(ParallelMode.TENSOR)),
        (ParallelMode.PIPELINE, parallel_context.get_local_rank(ParallelMode.PIPELINE)),
        (ParallelMode.DATA, parallel_context.get_local_rank(ParallelMode.DATA)),
    )
    device = parallel_context.ranks2device(ranks)

    assert isinstance(device, int)
    assert 0 <= device <= world_size


def test_device_mapping_in_parallel_context():
    TENSOR_PARALLEL_SIZE = 2
    PIPELINE_PARALLEL_SIZE = 2
    DATA_PARALLEL_SIZE = 2

    WORLD_SIZE = TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE * DATA_PARALLEL_SIZE

    SEED = 69
    BACKEND = "gloo"

    spawn(
        run_device_mapping_in_parallel_context,
        world_size=WORLD_SIZE,
        seed=SEED,
        backend=BACKEND,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
    )
