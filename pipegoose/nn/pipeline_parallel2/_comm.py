from queue import Queue

import torch.distributed.rpc as rpc


from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.pipeline_parallel2._package import Package

RECV_QUEUE = Queue()


def send_package(package: Package, src: int, dst: int, parallel_context: ParallelContext):
    assert isinstance(package, Package)

    dst_worker_name = parallel_context.get_worker_name(dst)
    rpc.rpc_sync(
        to=dst_worker_name,
        func=recv_package,
        args=(package, src, dst)
    )


def recv_package(package: Package, src: int, dst: int):
    # TODO: add configureable destination queue
    assert isinstance(package, Package)
    RECV_QUEUE.put(package)
