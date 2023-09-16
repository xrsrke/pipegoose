from copy import deepcopy

import pytest
import torch
from torch.optim import SGD
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.tensor_parallel.embedding import ParallelEmbedding
from pipegoose.nn.tensor_parallel.layer_norm import LayerNorm
from pipegoose.nn.tensor_parallel.linear import ColumnParallelLinear, RowParallelLinear
from pipegoose.nn.tensor_parallel.tensor_parallel import TensorParallel
from pipegoose.testing.utils import init_parallel_context, skip_in_github_actions, spawn

MODEL_NAME = "bigscience/bloom-560m"


@pytest.fixture(scope="session")
def model():
    return AutoModelForCausalLM.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def run_parallelize_a_transformers_and_inference(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, kwargs
):
    model = kwargs["model"]
    generation_configs = kwargs["generation_configs"]
    input = kwargs["input"]
    labels = kwargs["labels"]
    generated_tokens = kwargs["generated_tokens"]
    logits = kwargs["logits"]
    loss = kwargs["loss"]

    # NOTE: we don't parallelize dropout layers
    # and activation functions
    SKIP_MODULES = {type(model.transformer.h[0].mlp.gelu_impl), type(model.transformer.h[0].self_attention.attention_dropout)}

    def is_parallelized(module):
        return isinstance(module, (ParallelEmbedding, ColumnParallelLinear, RowParallelLinear, LayerNorm))

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
    assert torch.allclose(p_output.logits, logits, rtol=1e-1)
    assert torch.allclose(p_output.loss, loss, rtol=1e-1)


@skip_in_github_actions
@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4])
def test_parallelize_a_transformer_and_inference(model, tokenizer, tensor_parallel_size):
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    GENERATION_CONFIGS = {"max_new_tokens": 1}

    text = "Persistence is all you need."
    input = tokenizer(text, return_tensors="pt")
    labels = input["input_ids"]

    generated_tokens = model.generate(**input, **GENERATION_CONFIGS)
    outputs = model(**input, labels=labels)

    # NOTE: we make a copy of the model before updating its weights
    # so the output of the model is not affected by the updated weights
    orig_model = deepcopy(model)
    loss = outputs.loss
    logits = outputs.logits

    kwargs = {
        "model": orig_model,
        "generation_configs": GENERATION_CONFIGS,
        "input": input,
        "labels": labels,
        "generated_tokens": generated_tokens.detach(),
        "logits": logits.detach(),
        "loss": loss.detach(),
    }

    spawn(
        run_parallelize_a_transformers_and_inference,
        world_size=tensor_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        kwargs=kwargs,
    )


def run_backward_a_parallelized_transformers(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, kwargs
):
    model = kwargs["model"]
    lr = kwargs["lr"]
    input = kwargs["input"]
    labels = kwargs["labels"]
    embedding_weight = kwargs["embedding_weight"]

    def get_partition(data, dim, parallel_context):
        local_world_size = parallel_context.get_world_size(ParallelMode.TENSOR)
        local_rank = parallel_context.get_local_rank(ParallelMode.TENSOR)
        chunks = torch.chunk(data, chunks=local_world_size, dim=dim)
        return chunks[local_rank]

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    parallelized_model = TensorParallel(model, parallel_context).parallelize()

    p_output = parallelized_model(**input, labels=labels)
    p_loss = p_output.loss

    optim = SGD(parallelized_model.parameters(), lr=lr)
    optim.zero_grad()
    p_loss.backward()
    optim.step()

    # NOTE: our parallelized model only contains a partition of
    # the full weight, so we split the non-parallelized full weight and compare them
    p_embedding_weight = parallelized_model.transformer.word_embeddings.weight.data
    partitioned_embedding_weight = get_partition(embedding_weight, dim=0, parallel_context=parallel_context)
    # TODO: investigate why the rtol is so high
    assert torch.allclose(p_embedding_weight, partitioned_embedding_weight, rtol=1e-1)


@skip_in_github_actions
@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4])
def test_backward_pass_a_parallelized_transformers(model, tokenizer, tensor_parallel_size):
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    LR = 1e-3

    text = "Persistence is all you need."
    input = tokenizer(text, return_tensors="pt")
    labels = input["input_ids"]
    optim = SGD(model.parameters(), lr=LR)

    outputs = model(**input, labels=labels)

    # NOTE: we make a copy of the model before updating its weights
    # so the output of the model is not affected by the updated weights
    orig_model = deepcopy(model)
    loss = outputs.loss

    optim.zero_grad()
    loss.backward()
    optim.step()

    kwargs = {
        "model": orig_model,
        "lr": LR,
        "input": input,
        "labels": labels,
        # NOTE: this is the updated weight of the model
        "embedding_weight": model.transformer.word_embeddings.weight.data,
    }

    spawn(
        run_backward_a_parallelized_transformers,
        world_size=tensor_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        kwargs=kwargs,
    )
