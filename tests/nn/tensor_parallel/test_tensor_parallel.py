from copy import deepcopy

import pytest
import torch
from torch.optim import SGD
from transformers import AutoTokenizer, BloomConfig, BloomForCausalLM

from pipegoose.nn.tensor_parallel.embedding import ParallelEmbedding
from pipegoose.nn.tensor_parallel.layer_norm import LayerNorm
from pipegoose.nn.tensor_parallel.linear import ColumnParallelLinear, RowParallelLinear
from pipegoose.nn.tensor_parallel.tensor_parallel import TensorParallel
from pipegoose.testing.utils import init_parallel_context, spawn

MODEL_NAME = "bigscience/bloom-560m"


@pytest.fixture(scope="session")
def model():
    # NOTE: This model is similar to Bloom-560 in the architecture,
    # but smaller. We use it for fast unit tests.
    config = BloomConfig()
    model = BloomForCausalLM(config)
    return model


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def run_parallelize_a_transformers_and_inference(
    rank,
    world_size,
    port,
    tensor_parallel_size,
    pipeline_parallel_size,
    data_parallel_size,
    kwargs,
):
    def is_parallelized(module):
        return isinstance(
            module,
            (ParallelEmbedding, ColumnParallelLinear, RowParallelLinear, LayerNorm),
        )

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)

    def get_leaf_modules(model):
        leaf_modules = []
        for name, module in model.named_modules():
            if list(module.children()):
                continue
            leaf_modules.append((name, module))

        return leaf_modules

    model = deepcopy(kwargs["model"])
    generation_configs = kwargs["generation_configs"]
    input = kwargs["input"]
    REF_GENERATED_TOKENS = kwargs["generated_tokens"]

    # NOTE: we don't parallelize dropout layers
    # and activation functions
    SKIP_MODULES = {
        type(model.transformer.h[0].mlp.gelu_impl),
        type(model.transformer.h[0].self_attention.attention_dropout),
    }

    parallel_context = init_parallel_context(
        rank,
        world_size,
        port,
        tensor_parallel_size,
        pipeline_parallel_size,
        data_parallel_size,
    )

    parallelized_model = TensorParallel(model, parallel_context).parallelize()

    # NOTE: because pytorch also returns nested modules
    # and we only want to check the leaf modules,
    # so we filter out the nested modules
    leaf_modules = get_leaf_modules(parallelized_model)
    for module_name, module in leaf_modules:
        if type(module) in SKIP_MODULES:
            continue

        assert (
            is_parallelized(module) is True
        ), f"module {module_name} is not parallelized"

    generated_tokens = parallelized_model.generate(**input, **generation_configs)
    assert torch.allclose(generated_tokens, REF_GENERATED_TOKENS)


def test_data_parllel_fused_bias_gelu_bias_dropout_fwd():
    # TODO
    pass


@pytest.mark.parametrize("tensor_parallel_size", [2, 4])
def test_parallelize_a_transformer_and_inference(
    model, tokenizer, tensor_parallel_size
):
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    GENERATION_CONFIGS = {"max_new_tokens": 1}

    import random

    import numpy as np

    random.seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)

    text = "Persistence is all you need."
    inputs = tokenizer(text, return_tensors="pt")
    labels = torch.randint_like(inputs["input_ids"], low=100, high=200)

    generated_tokens = model.generate(**inputs, **GENERATION_CONFIGS)

    # NOTE: we make a copy of the model before updating its weights
    # so the output of the model is not affected by the updated weights
    orig_model = deepcopy(model)
    kwargs = {
        "model": orig_model,
        "generation_configs": GENERATION_CONFIGS,
        "input": inputs,
        "labels": labels,
        "generated_tokens": generated_tokens.detach(),
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
    rank,
    world_size,
    port,
    tensor_parallel_size,
    pipeline_parallel_size,
    data_parallel_size,
    kwargs,
):
    model = deepcopy(kwargs["model"])
    lr = kwargs["lr"]
    input = kwargs["input"]
    labels = kwargs["labels"]

    parallel_context = init_parallel_context(
        rank,
        world_size,
        port,
        tensor_parallel_size,
        pipeline_parallel_size,
        data_parallel_size,
    )

    parallelized_model = TensorParallel(model, parallel_context).parallelize()

    p_output = parallelized_model(**input, labels=labels)
    p_loss = p_output.loss

    optim = SGD(parallelized_model.parameters(), lr=lr)
    optim.zero_grad()
    p_loss.backward()
    optim.step()


@pytest.mark.parametrize("tensor_parallel_size", [2, 4])
def test_backward_pass_a_parallelized_transformers(
    model, tokenizer, tensor_parallel_size
):
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    LR = 1e-3

    text = "Persistence is all you need."
    input = tokenizer(text, return_tensors="pt")
    labels = input["input_ids"]

    kwargs = {
        "model": model,
        "lr": LR,
        "input": input,
        "labels": labels,
    }

    spawn(
        run_backward_a_parallelized_transformers,
        world_size=tensor_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        kwargs=kwargs,
    )
