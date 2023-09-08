import pytest
import torch
from torch.optim import SGD
from copy import deepcopy

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
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
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, model, generation_configs, input, labels, generated_tokens, logits, loss, updated_embedding_weights
):
    # NOTE: we don't parallelize dropout layers
    # and activation functions
    SKIP_MODULES = {
        type(model.transformer.h[0].mlp.gelu_impl),
        type(model.transformer.h[0].self_attention.attention_dropout)
    }
    LR = 1e-3

    def is_parallelized(module):
        return isinstance(module, (ParallelEmbedding, ColumnParallelLinear, RowParallelLinear, LayerNorm))

    def get_partition(data, dim, parallel_context):
        local_world_size = parallel_context.get_world_size(ParallelMode.TENSOR)
        local_rank = parallel_context.get_local_rank(ParallelMode.TENSOR)
        chunks = torch.chunk(data, chunks=local_world_size, dim=dim)
        return chunks[local_rank]

    def get_leaf_modules(model):
        leaf_modules = []
        for name, module in model.named_modules():
            if list(module.children()):
                continue
            leaf_modules.append((name, module))

        return leaf_modules

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

    p_generated_tokens = parallelized_model.generate(**input, **generation_configs)
    assert torch.allclose(p_generated_tokens, generated_tokens)

    p_output = parallelized_model(**input, labels=labels)
    p_loss = p_output.loss

    assert torch.allclose(p_output.logits, logits, rtol=1e-1)
    assert torch.allclose(p_loss, loss, rtol=1e-1)

    optim = SGD(parallelized_model.parameters(), lr=LR)
    optim.zero_grad()
    p_loss.backward()
    optim.step()

    assert torch.allclose(parallelized_model.transformer.word_embeddings.weight.data, get_partition(updated_embedding_weights, dim=0, parallel_context=parallel_context), rtol=1e-1)


@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4])
def test_parallelize_a_transformer(model, tokenizer, tensor_parallel_size):
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    GENERATION_CONFIGS = {
        "max_new_tokens": 1
    }
    LR = 1e-3

    text = "Persistence is all you need."
    input = tokenizer(text, return_tensors="pt")
    labels = input["input_ids"]
    optim = SGD(model.parameters(), lr=LR)

    generated_tokens = model.generate(**input, **GENERATION_CONFIGS)
    outputs = model(**input, labels=labels)
    BACKUP_MODEL = deepcopy(model)
    loss = outputs.loss
    logits = outputs.logits

    optim.zero_grad()
    loss.backward()
    optim.step()

    updated_embedding_weights = model.transformer.word_embeddings.weight.data

    spawn(
        run_parallelize_a_transformers,
        world_size=tensor_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        model=BACKUP_MODEL,
        generation_configs=GENERATION_CONFIGS,
        input=input,
        labels=labels,
        generated_tokens=generated_tokens.detach(),
        logits=logits.detach(),
        loss=loss.detach(),
        updated_embedding_weights=updated_embedding_weights.detach()
    )
