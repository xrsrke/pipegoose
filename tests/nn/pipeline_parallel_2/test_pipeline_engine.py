import time

import pytest
import torch
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.pipeline_parallel2._utils import sleep
from pipegoose.nn.pipeline_parallel2._worker import WorkerManager
from pipegoose.nn.pipeline_parallel2.pipeline_engine import PipelineEngine
from pipegoose.nn.pipeline_parallel2.queue import JobQueue
from pipegoose.nn.pipeline_parallel2.scheduler import GPipeScheduler
from pipegoose.testing.utils import init_parallel_context, spawn

model = nn.Sequential(
    nn.Linear(5, 5),
    nn.ReLU(),
    nn.Linear(5, 5),
)


def run_pipeline_engine(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, package):
    BATCH_SIZE = 32
    SEQ_LEN = 10
    HIDDEN_DIM = 5

    N_MICROBATCHES = 6

    inputs = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    scheduler = GPipeScheduler(N_MICROBATCHES, pipeline_parallel_size)
    # parallel_context = init_parallel_context(
    #     rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    # )
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

    forward_timeline = []

    class Function(nn.Module):
        def __init__(self, partition_idx):
            super().__init__()
            self.partition_idx = partition_idx
            self.microbatch_idx = 0
            self.net = nn.Linear(5, 5)

        def forward(self, input):
            time.sleep(0.5)
            forward_timeline.append((self.microbatch_idx, self.partition_idx))
            self.microbatch_idx += 1

            return self.net(input)

    worker_manager = WorkerManager()
    partition_func = Function(partition_idx=rank)
    pipeline_engine = PipelineEngine(
        module=model,
        scheduler=scheduler,
        rank=rank,
        worker_manager=worker_manager,
        parallel_context=parallel_context,
        partition_func=partition_func,
    )

    pipeline_engine.run(inputs)

    sleep(3)

    assert 1 == 1

    # if rank == 1:
    #     assert JobQueue.PENDING_JOBS.qsize() == 1

    # assert torch.allclose(outputs, model(inputs))


def run_test(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    from pipegoose.nn.pipeline_parallel2._utils import sleep

    QUEUE = []

    class FakeJob:
        def compute(self):
            QUEUE.append(1)

    job = FakeJob()
    WorkerManager().spawn()

    sleep(3)

    JobQueue.PENDING_JOBS.put(job)

    sleep(3)

    assert QUEUE == [1]


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2, 4])
def test_pipeline_engine(pipeline_parallel_size, forward_package):
    DATA_PARALLEL_SIZE = 1
    TENSOR_PARALLEL_SIZE = 1

    WORLD_SIZE = pipeline_parallel_size * DATA_PARALLEL_SIZE * TENSOR_PARALLEL_SIZE

    spawn(
        run_pipeline_engine,
        world_size=WORLD_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=DATA_PARALLEL_SIZE,
        package=forward_package,
    )
