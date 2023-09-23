from queue import Queue
from typing import Any

import torch.distributed.rpc as rpc

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.pipeline_parallel2._package import Package
from pipegoose.nn.pipeline_parallel2.pipeline_context import PipelineContext

RECV_QUEUE = Queue()
# RECV_QUEUE = dict()

# TODO: refactor to a singleton class
# NOTE: save parallel context for backward job
_PIPELINE_CONTEXT = None


def set_pipeline_context(pipeline_context: PipelineContext):
    global _PIPELINE_CONTEXT
    _PIPELINE_CONTEXT = pipeline_context


def get_pipeline_context() -> PipelineContext:
    return _PIPELINE_CONTEXT


def _send_data(data: Any, src: int, dst: int, parallel_context: ParallelContext):
    dst_worker_name = parallel_context.get_worker_name(dst)
    rpc.rpc_sync(to=dst_worker_name, func=_recv_package, args=(data, src, dst))


def send_package(package: Package, parallel_context: ParallelContext):
    """Send a package to another pipeline stage based on the metadata of the package."""

    assert isinstance(package, Package)

    rank = parallel_context.get_global_rank()

    if package.metadata.src == rank:
        dst = package.metadata.dst
        _send_data(package, src=rank, dst=dst, parallel_context=parallel_context)


def _recv_package(package: Package, src: int, dst: int):
    """
    Receive a package from another pipeline stage.

    NOTE: only be triggered by send_package.
    """
    # TODO: add configurable destination queue
    assert isinstance(package, Package)

    package.metadata.microbatch_idx
    package.metadata.partition_idx

    # TODO: refactor, user should not recall how to construct the key
    # RECV_QUEUE[(microbatch_idx, partition_idx)] = package
    RECV_QUEUE.put(package)
