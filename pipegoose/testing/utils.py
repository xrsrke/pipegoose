import os
import random
import socket
from functools import partial
from typing import Callable

import pytest
import torch.multiprocessing as mp

# NOTE: because these tests run too slow in GitHub Actions
skip_in_github_actions = pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Test skipped in GitHub Actions")


def find_free_port(min_port: int = 2000, max_port: int = 65000) -> int:
    while True:
        port = random.randint(min_port, max_port)
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("localhost", port))
                return port
        except OSError:
            continue


def spawn(func: Callable, world_size: int = 1, **kwargs):
    if kwargs.get("port") is None:
        port = find_free_port()
    else:
        port = kwargs["port"]
        kwargs.pop("port")

    wrapped_func = partial(func, world_size=world_size, port=port, **kwargs)
    mp.spawn(wrapped_func, nprocs=world_size)


def init_parallel_context(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    from pipegoose.distributed.parallel_context import ParallelContext

    HOST = "localhost"
    SEED = 69
    BACKEND = "gloo"

    parallel_context = ParallelContext(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        host=HOST,
        port=port,
        seed=SEED,
        backend=BACKEND,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )

    return parallel_context


N_PARTITIONS = 3
N_MICROBATCHES = 5


def init_pipeline_context(
    rank,
    world_size,
    port,
    tensor_parallel_size,
    pipeline_parallel_size,
    data_parallel_size,
    n_partitions=N_PARTITIONS,
    n_microbatches=N_MICROBATCHES,
):
    from torch import nn

    from pipegoose.nn.pipeline_parallel2.pipeline_context import PipelineContext
    from pipegoose.nn.pipeline_parallel2.scheduler import Scheduler, get_scheduler

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    partitions = [nn.Linear(10, 10) for _ in range(n_partitions)]
    scheduler = get_scheduler(Scheduler.GPIPE)(n_microbatches, n_partitions)

    return PipelineContext(
        partitions=partitions,
        scheduler=scheduler,
        parallel_context=parallel_context,
    )
