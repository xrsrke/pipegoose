import torch
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.tensor_parallel.linear import ParallelColumnLinear
from pipegoose.testing.utils import spawn


def init_parallel_context(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
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

    return parallel_context


def run_parallel_column_linear(
    rank,
    world_size,
    port,
    tensor_parallel_size,
    pipeline_parallel_size,
    data_parallel_size,
    batch_size,
    in_features,
    out_features,
    inputs,
    outputs,
    params,
    grads,
):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    local_rank = parallel_context.get_local_rank(parallel_mode=ParallelMode.TENSOR)
    ranks_in_group = parallel_context.get_ranks_in_group(parallel_mode=ParallelMode.TENSOR)

    if local_rank in ranks_in_group:
        local_world_size = parallel_context.get_world_size(parallel_mode=ParallelMode.TENSOR)

        model = ParallelColumnLinear(
            in_features,
            out_features,
            bias=True,
            gather_output=True,
            parallel_context=parallel_context,
        )

        partition_size = params["weights"].shape[0] // local_world_size
        partition_start, partition_end = local_rank * partition_size, (local_rank + 1) * partition_size

        model.weight.data = params["weights"][partition_start:partition_end, :]
        model.bias.data = params["biases"][partition_start:partition_end]

        parallel_outputs = model(inputs)

        assert parallel_outputs.shape == (batch_size, out_features)
        assert torch.allclose(parallel_outputs, outputs)


def test_parallel_column_linear():
    batch_size = 5
    in_features = 10
    out_features = 20

    inputs = torch.randn(batch_size, in_features)
    model = nn.Linear(in_features, out_features)

    outputs = model(inputs)
    outputs.sum().backward()

    params = {
        "weights": model.weight.detach().requires_grad_(False),
        "biases": model.bias.detach().requires_grad_(False),
    }

    grads = {
        "weights": model.weight.grad.detach().requires_grad_(False),
        "biases": model.bias.grad.detach().requires_grad_(False),
    }

    spawn(
        run_parallel_column_linear,
        world_size=8,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        data_parallel_size=2,
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
        inputs=inputs.detach(),
        outputs=outputs.detach(),
        params=params,
        grads=grads,
    )
