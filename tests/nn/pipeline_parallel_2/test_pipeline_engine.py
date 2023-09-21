import pytest
import torch
from torch import nn

from pipegoose.nn.pipeline_parallel2._worker import WorkerManager
from pipegoose.nn.pipeline_parallel2.pipeline_engine import PipelineEngine
from pipegoose.nn.pipeline_parallel2.scheduler import GPipeScheduler
from pipegoose.testing.utils import init_parallel_context, spawn

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
    worker_manager = WorkerManager()
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    pipeline_engine = PipelineEngine(
        module=model,
        scheduler=scheduler,
        worker_manager=worker_manager,
        parallel_context=parallel_context,
    )

    pipeline_engine.run(inputs)

    # assert torch.allclose(outputs, model(inputs))


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
def test_pipeline_engine(pipeline_parallel_size):
    DATA_PARALLEL_SIZE = 1
    TENSOR_PARALLEL_SIZE = 1

    spawn(
        run_pipeline_engine,
        world_size=pipeline_parallel_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=DATA_PARALLEL_SIZE,
    )
