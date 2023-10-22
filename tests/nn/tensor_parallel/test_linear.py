from copy import deepcopy

import pytest
import torch
from torch import nn

from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.tensor_parallel.linear import ColumnParallelLinear, RowParallelLinear
from pipegoose.testing.utils import get_partition, init_parallel_context, spawn


def run_column_parallel_linear(
    rank,
    world_size,
    port,
    tensor_parallel_size,
    pipeline_parallel_size,
    data_parallel_size,
    in_features,
    out_features,
    inputs,
    ref_outputs,
    orig_params,
    ref_params,
    ref_grads,
):
    ORIG_PARAMS = deepcopy(orig_params)
    REF_PARAMS = deepcopy(ref_params)
    REF_GRADS = deepcopy(ref_grads)

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    local_rank = parallel_context.get_local_rank(parallel_mode=ParallelMode.TENSOR)
    ranks_in_group = parallel_context.get_ranks_in_group(parallel_mode=ParallelMode.TENSOR)

    if local_rank in ranks_in_group:
        local_world_size = parallel_context.get_world_size(parallel_mode=ParallelMode.TENSOR)

        model = ColumnParallelLinear(
            in_features,
            out_features,
            bias=True,
            gather_output=True,
            parallel_context=parallel_context,
        )

        partition_size = ORIG_PARAMS["weight"].shape[0] // local_world_size
        partition_start, partition_end = local_rank * partition_size, (local_rank + 1) * partition_size

        model.weight.data = ORIG_PARAMS["weight"][partition_start:partition_end, :]
        model.bias.data = ORIG_PARAMS["bias"][partition_start:partition_end]

        outputs = model(inputs)

        assert outputs.shape == ref_outputs.shape
        assert torch.allclose(outputs, ref_outputs)

        outputs.sum().backward()

        split_dim = 0
        REF_WEIGHT_GRADS = get_partition(REF_GRADS["weight"], dim=split_dim, parallel_context=parallel_context)
        REF_BIAS_GRADS = get_partition(REF_GRADS["bias"], dim=split_dim, parallel_context=parallel_context)
        assert torch.allclose(model.weight.grad, REF_WEIGHT_GRADS)
        assert torch.allclose(model.bias.grad, REF_BIAS_GRADS)

        REF_WEIGHT = get_partition(REF_PARAMS["weight"], dim=split_dim, parallel_context=parallel_context)
        REF_BIAS = get_partition(REF_PARAMS["bias"], dim=split_dim, parallel_context=parallel_context)
        assert torch.allclose(model.weight, REF_WEIGHT)
        assert torch.allclose(model.bias, REF_BIAS)


def run_row_parallel_linear(
    rank,
    world_size,
    port,
    tensor_parallel_size,
    pipeline_parallel_size,
    data_parallel_size,
    in_features,
    out_features,
    inputs,
    ref_outputs,
    orig_params,
    ref_params,
    ref_grads,
):
    ORIG_PARAMS = deepcopy(orig_params)
    REF_PARAMS = deepcopy(ref_params)
    REF_GRADS = deepcopy(ref_grads)

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    local_rank = parallel_context.get_local_rank(parallel_mode=ParallelMode.TENSOR)
    ranks_in_group = parallel_context.get_ranks_in_group(parallel_mode=ParallelMode.TENSOR)

    if local_rank in ranks_in_group:
        local_world_size = parallel_context.get_world_size(parallel_mode=ParallelMode.TENSOR)

        model = RowParallelLinear(
            in_features,
            out_features,
            bias=True,
            parallel_context=parallel_context,
        )

        partition_size = ORIG_PARAMS["weight"].shape[1] // local_world_size
        partition_start, partition_end = local_rank * partition_size, (local_rank + 1) * partition_size

        model.weight.data = ORIG_PARAMS["weight"][:, partition_start:partition_end]
        model.bias.data = ORIG_PARAMS["bias"]

        outputs = model(inputs)

        assert outputs.shape == ref_outputs.shape
        assert torch.allclose(outputs, ref_outputs)

        outputs.sum().backward()

        split_dim = 1
        REF_WEIGHT_GRADS = get_partition(REF_GRADS["weight"], dim=split_dim, parallel_context=parallel_context)
        assert torch.allclose(model.weight.grad, REF_WEIGHT_GRADS)
        assert torch.allclose(model.bias.grad, REF_GRADS["bias"])

        REF_WEIGHT = get_partition(REF_PARAMS["weight"], dim=split_dim, parallel_context=parallel_context)
        assert torch.allclose(model.weight, REF_WEIGHT)
        assert torch.allclose(model.bias, REF_PARAMS["bias"])


@pytest.mark.parametrize("run_linear", [run_column_parallel_linear, run_row_parallel_linear])
def test_parallel_linear(run_linear):
    TENSOR_PARALLEL_SIZE = 2
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    batch_size = 5
    in_features = 10
    out_features = 20

    inputs = torch.randn(batch_size, in_features)
    model = nn.Linear(in_features, out_features)
    ORIG_PARAMS = {
        "weight": deepcopy(model.weight.detach().requires_grad_(False)),
        "bias": deepcopy(model.bias.detach().requires_grad_(False)),
    }

    outputs = model(inputs)
    outputs.sum().backward()

    REF_GRADS = {
        "weight": model.weight.grad.detach().requires_grad_(False),
        "bias": model.bias.grad.detach().requires_grad_(False),
    }
    REF_PARAMS = {
        "weight": deepcopy(model.weight.detach().requires_grad_(False)),
        "bias": deepcopy(model.bias.detach().requires_grad_(False)),
    }

    spawn(
        run_linear,
        world_size=TENSOR_PARALLEL_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        in_features=in_features,
        out_features=out_features,
        inputs=inputs.detach(),
        ref_outputs=outputs.detach(),
        orig_params=ORIG_PARAMS,
        ref_params=REF_PARAMS,
        ref_grads=REF_GRADS,
    )
