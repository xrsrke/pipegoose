import time

import pytest

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.pipeline_parallel2._comm import RECV_QUEUE, send_package
from pipegoose.nn.pipeline_parallel2._package import Package
from pipegoose.testing.utils import spawn


def init_parallel_context(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    parallel_context = ParallelContext(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        host="localhost",
        port=port,
        seed=69,
        backend="gloo",
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )

    return parallel_context


def run_send_recv_package(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, package):
    PACKAGE_SRC_RANK = package.metadata.src
    PACKAGE_DST_RANK = package.metadata.dst

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    if rank == PACKAGE_SRC_RANK:
        send_package(package, parallel_context=parallel_context)
    elif rank == PACKAGE_DST_RANK:
        time.sleep(1)
        received_package = RECV_QUEUE.get()

        assert isinstance(received_package, Package)


@pytest.mark.parametrize("pipeline_parallel_size", [2])
def test_run_send_recv_package(forward_package, pipeline_parallel_size):
    TENSOR_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    spawn(
        run_send_recv_package,
        world_size=pipeline_parallel_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=DATA_PARALLEL_SIZE,
        package=forward_package,
    )
