import time

import pytest

from pipegoose.nn.pipeline_parallel2._comm import RECV_QUEUE, send_package
from pipegoose.nn.pipeline_parallel2._package import Package
from pipegoose.testing.utils import init_parallel_context, spawn


def run_send_recv_package(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, package):
    PACKAGE_SRC_RANK = package.metadata.src
    PACKAGE_DST_RANK = package.metadata.dst

    # MICROBATCH_IDX = package.metadata.microbatch_idx
    # PARTITION_IDX = package.metadata.partition_idx

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    if rank == PACKAGE_SRC_RANK:
        send_package(package, parallel_context=parallel_context)
    elif rank == PACKAGE_DST_RANK:
        time.sleep(1)
        received_package = RECV_QUEUE.get()
        # received_package = RECV_QUEUE[(MICROBATCH_IDX, PARTITION_IDX)]

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
