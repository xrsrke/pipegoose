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
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, kwargs
):
    model = kwargs["model"]
    generation_configs = kwargs["generation_configs"]
    lr = kwargs["lr"]
    input = kwargs["input"]
    labels = kwargs["labels"]
    generated_tokens = kwargs["generated_tokens"]
    logits = kwargs["logits"]
    loss = kwargs["loss"]
    embedding_weight = kwargs["embedding_weight"]

    # NOTE: we don't parallelize dropout layers
    # and activation functions
    SKIP_MODULES = {
        type(model.transformer.h[0].mlp.gelu_impl),
        type(model.transformer.h[0].self_attention.attention_dropout)
    }

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

    optim = SGD(parallelized_model.parameters(), lr=lr)
    optim.zero_grad()
    p_loss.backward()
    optim.step()

    # NOTE: our parallelized model only contains a partition of
    # the full weight, so we split the full weight and compare them
    p_embedding_weight = parallelized_model.transformer.word_embeddings.weight.data
    partitioned_updated_weight = get_partition(embedding_weight, dim=0, parallel_context=parallel_context)
    assert torch.allclose(p_embedding_weight, partitioned_updated_weight, rtol=1e-3)


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

    # NOTE: we make a copy of the model before updating its weights
    # so the output of the model is not affected by the updated weights
    orig_model = deepcopy(model)
    loss = outputs.loss
    logits = outputs.logits

    optim.zero_grad()
    loss.backward()
    optim.step()

    kwargs = {
        "model": orig_model,
        "generation_configs": GENERATION_CONFIGS,
        "lr": LR,
        "input": input,
        "labels": labels,
        "generated_tokens": generated_tokens.detach(),
        "logits": logits.detach(),
        "loss": loss.detach(),
        # NOTE: this is the updated weight of the model
        "embedding_weight": model.transformer.word_embeddings.weight.data
    }

    spawn(
        run_parallelize_a_transformers,
        world_size=tensor_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        kwargs=kwargs
    )
