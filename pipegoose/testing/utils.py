import os
import random
import socket
from functools import partial
from typing import Callable

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode

# NOTE: because these tests run too slow in GitHub Actions
skip_in_github_actions = pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Test skipped in GitHub Actions")
skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def find_free_port(min_port: int = 2000, max_port: int = 65000) -> int:
    while True:
        port = random.randint(min_port, max_port)
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("localhost", port))
                return port
        except OSError as e:
            raise e


def spawn(func: Callable, world_size: int = 1, **kwargs):
    if kwargs.get("port") is None:
        port = find_free_port()
    else:
        port = kwargs["port"]
        kwargs.pop("port")

    wrapped_func = partial(func, world_size=world_size, port=port, **kwargs)
    mp.spawn(wrapped_func, nprocs=world_size)


def init_parallel_context(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
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
    from pipegoose.nn.pipeline_parallel.pipeline_context import PipelineContext
    from pipegoose.nn.pipeline_parallel.scheduler import SchedulerType, get_scheduler

    N_PARTITIONS = pipeline_parallel_size
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    scheduler = get_scheduler(SchedulerType.GPIPE)(n_microbatches, N_PARTITIONS)
    pipeline_context = PipelineContext(
        scheduler=scheduler,
        parallel_context=parallel_context,
    )

    return pipeline_context, parallel_context


def get_partition(data: torch.Tensor, dim: int, parallel_context: ParallelContext) -> torch.Tensor:
    local_world_size = parallel_context.get_world_size(ParallelMode.TENSOR)
    local_rank = parallel_context.get_local_rank(ParallelMode.TENSOR)
    chunks = torch.chunk(data, chunks=local_world_size, dim=dim)
    return chunks[local_rank]


def calculate_parameter_similarity(module1: nn.Module, module2: nn.Module, rtol: float = 1e-3) -> float:
    # NOTE: In some test cases, the parameters of an updated model after
    # .step() are very close to the parameters of the original model.
    # So we use this function to check if the parameters of
    # the updated model have deviated from the parameters
    # of the original model enough.
    total_parameters, equal_parameters = 0, 0
    for param1, param2 in zip(module1.parameters(), module2.parameters()):
        assert param1.shape == param2.shape, "Parameters have different shapes"
        flat_param1, flat_param2 = param1.view(-1), param2.view(-1)
        total_parameters += flat_param1.shape[0]
        equal_parameters += torch.sum(torch.isclose(flat_param1, flat_param2, rtol=rtol)).item()

    return equal_parameters / total_parameters
