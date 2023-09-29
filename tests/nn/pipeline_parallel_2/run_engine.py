import time

import torch
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.pipeline_parallel2._utils import get_partition_idx, sleep
from pipegoose.nn.pipeline_parallel2._worker import WorkerManager
from pipegoose.nn.pipeline_parallel2.pipeline_engine import PipelineEngine
from pipegoose.nn.pipeline_parallel2.scheduler import GPipeScheduler
from pipegoose.testing.utils import spawn

model = nn.Sequential(
    nn.Linear(5, 5),
    nn.ReLU(),
    nn.Linear(5, 5),
)


def run_pipeline_engine(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
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
    partition_idx = get_partition_idx(parallel_context)
    partition_func = Function(partition_idx)
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

    assert forward_timeline == [
        (0, partition_idx),
        (1, partition_idx),
        (2, partition_idx),
        (3, partition_idx),
        (4, partition_idx),
    ]


if __name__ == "__main__":
    DATA_PARALLEL_SIZE = 1
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 4

    WORLD_SIZE = PIPELINE_PARALLEL_SIZE * DATA_PARALLEL_SIZE * TENSOR_PARALLEL_SIZE

    # from hanging_threads import start_monitoring
    # monitoring_thread = start_monitoring()

    spawn(
        run_pipeline_engine,
        world_size=WORLD_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
    )
