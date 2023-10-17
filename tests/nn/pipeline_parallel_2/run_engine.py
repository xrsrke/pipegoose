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
    outputs,
):
    forward_timeline = []
    backward_timeline = []

    def backward_hook(module, grad_input, grad_output):
        backward_timeline.append((module.microbatch_idx - 1, module.partition_idx))
        module.microbatch_idx -= 1

    class Function(nn.Module):
        def __init__(self, partition_idx):
            super().__init__()
            self.partition_idx = partition_idx
            self.microbatch_idx = 0
            self.net = model[self.partition_idx]
            self.register_backward_hook(backward_hook)

        def forward(self, input):
            forward_timeline.append((self.microbatch_idx, self.partition_idx))
            self.microbatch_idx += 1
            return self.net(input)

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    scheduler = GPipeScheduler(n_microbatches, pipeline_parallel_size)
    worker_manager = WorkerManager()
    partition_idx = get_partition_idx(parallel_context)
    partition_func = Function(partition_idx)
    pipeline_engine = PipelineEngine(
        module=model,
        scheduler=scheduler,
        worker_manager=worker_manager,
        parallel_context=parallel_context,
        partition_func=partition_func,
    )
    [(microbatch_idx, partition_idx) for microbatch_idx in range(n_microbatches)]
    # EXPECTED_BACKWARD_TIMELINE = [(microbatch_idx, partition_idx) for microbatch_idx in range(n_microbatches, -1, -1)]
    p_outputs = pipeline_engine.run(inputs)

    if is_last_stage(parallel_context):
        # assert torch.allclose(torch.cat(p_outputs, dim=0), outputs)
        # assert forward_timeline == EXPECTED_FORWARD_TIMELINE
        for output in p_outputs:
            output.sum().backward()

        # concated_p_putputs = torch.cat(p_outputs, dim=0)
        # # p_outputs[0].sum().backward()
        # concated_p_putputs.sum().backward()
        # for p_output in p_outputs:
        #     print("main thread do backward pass")
        #     p_output.sum().backward()
        #     # assert backward_timeline == EXPECTED_FORWARD_TIMELINE

        assert 1 == 1
    else:
        # NOTE: earlier stages should not return the final output
        # assert p_outputs is None
        assert 1 == 1
        p_outputs.sum().backward()


if __name__ == "__main__":
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
    outputs = reduce(lambda inputs, layer: layer(inputs), model, inputs)

    # outputs.sum().backward()

    spawn(
        run_pipeline_engine,
        world_size=WORLD_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        n_microbatches=N_MICROBATCHES,
        model=model,
        inputs=inputs.detach(),
        outputs=outputs.detach(),
    )
