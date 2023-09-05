import pytest
import torch

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.tensor_parallel.tensor_parallel import TensorParallel
from pipegoose.nn.tensor_parallel.embedding import ParallelEmbedding
from pipegoose.nn.tensor_parallel.linear import ColumnParallelLinear, RowParallelLinear
from pipegoose.nn.tensor_parallel.layer_norm import LayerNorm
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


def run_parallelize_a_transformers(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, model, input, output
):
    # NOTE: we don't parallelize dropout layers
    # and activation functions
    SKIP_MODULES = {
        type(model.transformer.h[0].mlp.gelu_impl),
        type(model.transformer.h[0].self_attention.attention_dropout)
    }

    def get_leaf_modules(model):
        leaf_modules = []
        for name, module in model.named_modules():
            if list(module.children()):
                continue
            leaf_modules.append((name, module))

        return leaf_modules

    def is_parallelized(module):
        return isinstance(module, (ParallelEmbedding, ColumnParallelLinear, RowParallelLinear, LayerNorm))

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    parallelized_model = TensorParallel(model, parallel_context).parallelize()

    # NOTE: because pytorch also returns nested modules
    # and we only want to check the leaf modules,
    # so we filter out the nested modules
    leaf_modules = get_leaf_modules(parallelized_model)
    for module_name, module in leaf_modules:
        if type(module) in SKIP_MODULES:
            continue

        assert is_parallelized(module) is True, f"module {module_name} is not parallelized"

    if world_size == 1:
        # NOTE: If the world_size is larger than 1,
        # we have to partition the weights and load them after
        # parallelizing a module. This is because if process 1 splits
        # a weight in half and reassigns it, and then process 2
        # also splits and reassigns the weight, it will result in the weight
        # being split into 4 parts. This isn't what we want;
        # we want the weight to be split into 2 parts.
        # so we are moving this to another test.
        p_output = parallelized_model.generate(**input)
        assert torch.allclose(p_output, output)


@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
def test_parallelize_a_transformer(model, tokenizer, tensor_parallel_size):
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    text = "Persistence is all you need."
    input = tokenizer(text, return_tensors="pt")
    output = model.generate(**input)

    spawn(
        run_parallelize_a_transformers,
        world_size=tensor_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        model=model,
        input=input,
        output=output.detach(),
    )
