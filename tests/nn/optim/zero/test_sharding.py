from copy import deepcopy

import pytest
from torch import nn
from torch.optim import SGD
from transformers import AutoModel

from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.optim.zero.sharding import OptimizerStateSharding
from pipegoose.testing.utils import init_parallel_context, spawn


def run_optimizer_states_sharding(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, model
):
    def calculate_total_sharded_elements(sharded_params):
        total = 0
        num_params_per_partition = []
        for param_groups in sharded_params:
            local_total = 0
            for param_group in param_groups:
                for param in param_group["params"]:
                    local_total += param.numel()

            num_params_per_partition.append(local_total)
            total += local_total
        return total, num_params_per_partition

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    world_size = parallel_context.get_world_size(ParallelMode.DATA)

    ORIG_MODEL = deepcopy(model)
    optim = SGD(model.parameters(), lr=0.01)
    param_groups = optim.param_groups

    sharder = OptimizerStateSharding(param_groups, parallel_context, ParallelMode.DATA)
    sharded_params = sharder.shard()

    assert len(sharded_params) == world_size

    for rank, shard in enumerate(sharded_params):
        if rank == 4:
            assert 1 == 1

        assert isinstance(shard, list)
        for param_group in shard:
            assert len(param_group["params"]) > 0
            for param in param_group["params"]:
                assert isinstance(param, nn.Parameter)

        # NOTE: each rank, expect to have the same number of parameter groups
        assert len(shard) == len(optim.param_groups)

    total_elements = sum(param.numel() for param in ORIG_MODEL.parameters())
    total_sharded_elements, num_params_per_partition = calculate_total_sharded_elements(sharded_params)
    assert total_sharded_elements == total_elements
    # NOTE: each partition, expect to have less than the total number of parameters
    for num_param in num_params_per_partition:
        assert num_param < total_elements


@pytest.mark.parametrize("model_name", ["torch_module", "transformer"])
@pytest.mark.parametrize("data_parallel_size", [2, 5])
def test_optimizer_states_sharding(model_name, data_parallel_size):
    if model_name == "torch_module":
        INPUT_SIZE = 5
        HIDDEN_SIZE = 10
        OUTPUT_SIZE = 2
        model = nn.Sequential(nn.Linear(INPUT_SIZE, HIDDEN_SIZE), nn.ReLU(), nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE))
    else:
        model = AutoModel.from_pretrained("gpt2")

    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 1
    WORLD_SIZE = TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE * data_parallel_size

    spawn(
        run_optimizer_states_sharding,
        world_size=WORLD_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=data_parallel_size,
        model=model,
    )
