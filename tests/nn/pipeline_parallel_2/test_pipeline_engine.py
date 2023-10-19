from copy import deepcopy
from functools import reduce

import torch
from torch import nn

from pipegoose.nn.pipeline_parallel2._utils import get_partition_idx, is_last_stage
from pipegoose.nn.pipeline_parallel2._worker import WorkerManager
from pipegoose.nn.pipeline_parallel2.pipeline_engine import PipelineEngine
from pipegoose.nn.pipeline_parallel2.scheduler import GPipeScheduler
from pipegoose.testing.utils import init_parallel_context, spawn


def run_pipeline_engine(
    rank,
    world_size,
    port,
    tensor_parallel_size,
    pipeline_parallel_size,
    data_parallel_size,
    n_microbatches,
    model,
    inputs,
    ref_outputs,
    ref_grads,
):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    partition_idx = get_partition_idx(parallel_context)
    partition = model[partition_idx]
    scheduler = GPipeScheduler(n_microbatches, pipeline_parallel_size)
    worker_manager = WorkerManager()
    pipeline_engine = PipelineEngine(
        module=partition,
        scheduler=scheduler,
        worker_manager=worker_manager,
        parallel_context=parallel_context,
    )
    outputs = pipeline_engine.run(inputs)

    if is_last_stage(parallel_context):
        assert torch.allclose(torch.cat(outputs, dim=0), ref_outputs)

    for output in outputs:
        output.sum().backward(retain_graph=True)

    for p, ref_grad in zip(partition.parameters(), ref_grads[partition_idx]):
        assert p.grad is not None
        assert torch.allclose(p.grad, ref_grad)


def test_pipeline_engine():
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 4
    DATA_PARALLEL_SIZE = 1

    BATCH_SIZE = 32
    N_MICROBATCHES = 6
    SEQ_LEN = 10
    HIDDEN_DIM = 5
    WORLD_SIZE = TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE * DATA_PARALLEL_SIZE

    inputs = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, requires_grad=False)
    model = nn.ModuleList([nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU()) for _ in range(PIPELINE_PARALLEL_SIZE)])
    ORIG_MODEL = deepcopy(model)
    outputs = reduce(lambda inputs, layer: layer(inputs), model, inputs)

    outputs.sum().backward()

    grads = [[p.grad for p in layer.parameters()] for layer in model]

    spawn(
        run_pipeline_engine,
        world_size=WORLD_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        n_microbatches=N_MICROBATCHES,
        model=ORIG_MODEL,
        inputs=inputs.detach(),
        ref_outputs=outputs.detach(),
        ref_grads=grads,
    )
