from copy import deepcopy
from functools import reduce

import pytest
import torch
from torch import nn

from pipegoose.nn.pipeline_parallel2._utils import is_last_stage
from pipegoose.nn.pipeline_parallel2.pipeline_parallel import PipelineParallel
from pipegoose.testing.utils import init_parallel_context, spawn


def run_pipeline_parallel(
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
):

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    model = PipelineParallel(model, num_microbatches=n_microbatches, parallel_context=parallel_context).parallelize()

    outputs = model(inputs)

    if is_last_stage(parallel_context):
        assert torch.allclose(torch.cat(outputs, dim=0), ref_outputs)

    for output in outputs:
        output.sum().backward(retain_graph=True)


@pytest.mark.parametrize(
    "tensor_parallel_size, pipeline_parallel_size, data_parallel_size",
    [
        (1, 4, 1),
        # TODO: not works with 3d parallelism yet
        # (2, 4, 2)
    ],
)
def test_pipeline_parallel(tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    BATCH_SIZE = 32
    N_MICROBATCHES = 6
    SEQ_LEN = 10
    HIDDEN_DIM = 5
    WORLD_SIZE = tensor_parallel_size * pipeline_parallel_size * data_parallel_size

    inputs = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, requires_grad=False)
    model = nn.ModuleList([nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU()) for _ in range(pipeline_parallel_size)])
    ORIG_MODEL = deepcopy(model)
    outputs = reduce(lambda inputs, layer: layer(inputs), model, inputs)

    outputs.sum().backward()

    spawn(
        run_pipeline_parallel,
        world_size=WORLD_SIZE,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        n_microbatches=N_MICROBATCHES,
        model=ORIG_MODEL,
        inputs=inputs.detach(),
        ref_outputs=outputs.detach(),
    )
