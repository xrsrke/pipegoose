import pytest

from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel2.sync.handshake import ParallelGroupHandshake
from pipegoose.testing.utils import init_parallel_context, spawn


def run_parallel_group_handshake(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, parallel_mode
):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    handshake = ParallelGroupHandshake(
        parallel_context,
        parallel_mode=parallel_mode,
    )
    handshake.initiate()
    handshake.confirm()
    handshake.barrier()


@pytest.mark.parametrize(
    "parallel_mode, tensor_parallel_size, pipeline_paralell_size, data_parallel_size",
    [
        (ParallelMode.GLOBAL, 2, 2, 2),
        (ParallelMode.TENSOR, 2, 1, 1),
        (ParallelMode.PIPELINE, 1, 2, 1),
        (ParallelMode.DATA, 1, 1, 2),
    ],
)
def test_parallel_group_handshake(parallel_mode, tensor_parallel_size, pipeline_paralell_size, data_parallel_size):
    world_size = tensor_parallel_size * pipeline_paralell_size * data_parallel_size

    spawn(
        run_parallel_group_handshake,
        world_size=world_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_paralell_size,
        data_parallel_size=data_parallel_size,
        parallel_mode=parallel_mode,
    )
