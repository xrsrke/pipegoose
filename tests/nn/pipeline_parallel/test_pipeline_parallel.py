from copy import deepcopy
from functools import reduce

import torch
from torch import nn

from pipegoose.nn.pipeline_parallel._utils import get_partition_idx, is_last_stage
from pipegoose.nn.pipeline_parallel.pipeline_parallel import PipelineParallel
from pipegoose.testing.utils import init_parallel_context, spawn


def generate_expected_timeline(num_microbatches, partition_idx):
    # NOTE: example: [(5, 2), (4, 2), (3, 2), (2, 2), (1, 2), (0, 2)]
    # (x, y) where x is microbatch_idx, and y is partition_idx
    forward_timeline = [(microbatch_idx, partition_idx) for microbatch_idx in range(num_microbatches)]
    backward_timeline = [(microbatch_idx, partition_idx) for microbatch_idx in range(num_microbatches - 1, -1, -1)]
    return forward_timeline, backward_timeline


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def run_pipeline_parallel(
    rank,
    world_size,
    port,
    tensor_parallel_size,
    pipeline_parallel_size,
    data_parallel_size,
    num_microbatches,
    model,
    inputs,
    ref_outputs,
    ref_grads,
):
    forward_timeline = []
    backward_timeline = []

    def backward_hook(module, grad_input, grad_output):
        if module.microbatch_idx > 0:
            backward_timeline.append((module.microbatch_idx - 1, module.partition_idx))
            module.microbatch_idx -= 1

    class TimelineRegister(nn.Module):
        def __init__(self, partition_idx, module):
            super().__init__()
            self.module = module
            self.module.partition_idx = partition_idx
            self.module.microbatch_idx = 0
            self.module.register_backward_hook(backward_hook)

        def forward(self, input):
            forward_timeline.append((self.module.microbatch_idx, self.module.partition_idx))
            self.module.microbatch_idx += 1
            return self.module(input)

    # NOTE: just record the forward and backward timeline
    model = nn.ModuleList([TimelineRegister(partition_idx, module) for partition_idx, module in enumerate(model)])

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    partition_idx = get_partition_idx(parallel_context)
    EXPECTED_FORWARD_TIMELINE, EXPECTED_BACKWARD_TIMELINE = generate_expected_timeline(num_microbatches, partition_idx)

    parallelized_model = PipelineParallel(
        model, num_microbatches=num_microbatches, parallel_context=parallel_context
    ).parallelize()

    assert isinstance(parallelized_model, nn.Module)
    assert count_parameters(parallelized_model) < count_parameters(model)
    assert count_parameters(parallelized_model) == count_parameters(model[partition_idx])

    outputs = parallelized_model(inputs)

    assert forward_timeline == EXPECTED_FORWARD_TIMELINE
    if is_last_stage(parallel_context):
        assert torch.allclose(torch.cat(outputs, dim=0), ref_outputs)

    for output in outputs:
        output.sum().backward(retain_graph=True)

    assert backward_timeline == EXPECTED_BACKWARD_TIMELINE
    for p, ref_grad in zip(parallelized_model.parameters(), ref_grads[partition_idx]):
        assert p.grad is not None
        assert torch.allclose(p.grad, ref_grad)


def test_pipeline_parallel():
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 4
    DATA_PARALLEL_SIZE = 1

    BATCH_SIZE = 32
    NUM_MICROBATCHES = 6
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
        run_pipeline_parallel,
        world_size=WORLD_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        num_microbatches=NUM_MICROBATCHES,
        model=ORIG_MODEL,
        inputs=inputs.detach(),
        ref_outputs=outputs.detach(),
        ref_grads=grads,
    )
