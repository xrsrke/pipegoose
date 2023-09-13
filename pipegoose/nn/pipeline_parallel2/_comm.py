from typing import Any
from queue import Queue

import torch.distributed.rpc as rpc


from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.pipeline_parallel2._package import Package

RECV_QUEUE = Queue()


def _send_data(data: Any, src: int, dst: int, parallel_context: ParallelContext):
    dst_worker_name = parallel_context.get_worker_name(dst)
    rpc.rpc_sync(
        to=dst_worker_name,
        func=recv_data,
        args=(data, src, dst)
    )


def send_package(package: Package, parallel_context: ParallelContext):
    """Send a package to another pipeline stage based on the metadata of the package."""

    assert isinstance(package, Package)

    rank = parallel_context.get_global_rank()

    if package.metadata.src == rank:
        dst = package.metadata.dst
        _send_data(package, src=rank, dst=dst, parallel_context=parallel_context)


def recv_data(package: Package, src: int, dst: int):
    # TODO: add configureable destination queue
    assert isinstance(package, Package)
    RECV_QUEUE.put(package)
