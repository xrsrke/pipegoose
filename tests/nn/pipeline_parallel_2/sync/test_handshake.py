from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel2.sync.handshake import ParallelGroupHandshake
from pipegoose.testing.utils import init_parallel_context, spawn


def run_parallel_group_handshake(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    PARALLEL_MODE = ParallelMode.GLOBAL

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    handshake = ParallelGroupHandshake(
        parallel_context,
        parallel_mode=PARALLEL_MODE,
    )
    handshake.initiate()
    handshake.confirm()
    handshake.barrier()


def test_parallel_group_handshake():
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 3
    DATA_PARALLEL_SIZE = 1
    world_size = TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE * DATA_PARALLEL_SIZE

    spawn(
        run_parallel_group_handshake,
        world_size=world_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
    )
