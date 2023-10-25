import random
import time

import pytest
import torch

from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel.sync.handshake import ParallelGroupHandshake
from pipegoose.testing.utils import init_parallel_context, spawn


def run_parallel_group_handshake(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, parallel_mode, shared_counter
):
    def do_random_delay():
        rand_time = random.uniform(0, 3)
        time.sleep(rand_time)

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    # NOTE: simulate some random delay in different ranks
    # before the handshake
    do_random_delay()

    handshake = ParallelGroupHandshake(
        parallel_context,
        parallel_mode=parallel_mode,
    )
    handshake.initiate()

    do_random_delay()
    handshake.confirm()
    shared_counter.add_(rank)

    handshake.barrier()

    # NOTE: since each process adds its rank to the shared counter, the sum of all ranks should be equal to
    local_world_size = parallel_context.get_world_size(parallel_mode)
    assert shared_counter.item() == sum(x for x in range(local_world_size))


@pytest.mark.parametrize(
    "parallel_mode, tensor_parallel_size, pipeline_paralell_size, data_parallel_size",
    [
        # (ParallelMode.GLOBAL, 2, 2, 2),
        # (ParallelMode.TENSOR, 2, 1, 1),
        (ParallelMode.PIPELINE, 1, 2, 1),
        # (ParallelMode.DATA, 1, 1, 2),
    ],
)
def test_parallel_group_handshake(parallel_mode, tensor_parallel_size, pipeline_paralell_size, data_parallel_size):
    WORLD_SIZE = tensor_parallel_size * pipeline_paralell_size * data_parallel_size

    shared_counter = torch.tensor(0)
    shared_counter.share_memory_()

    spawn(
        run_parallel_group_handshake,
        world_size=WORLD_SIZE,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_paralell_size,
        data_parallel_size=data_parallel_size,
        parallel_mode=parallel_mode,
        shared_counter=shared_counter,
    )
