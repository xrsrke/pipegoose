from copy import deepcopy

import pytest
import torch
from torch.optim import SGD
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipegoose.nn.tensor_parallel.embedding import ParallelEmbedding
from pipegoose.nn.tensor_parallel.linear import ColumnParallelLinear, RowParallelLinear
from pipegoose.nn.tensor_parallel.tensor_parallel import TensorParallel
from pipegoose.testing.utils import (
    get_partition,
    init_parallel_context,
    skip_in_github_actions,
    spawn,
)

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
    def is_parallelized(module):
        # LayerNorm
        return isinstance(module, (ParallelEmbedding, ColumnParallelLinear, RowParallelLinear))

    import random

    import numpy as np

    random.seed(42)
    np.random.seed(42)
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
    labels = kwargs["labels"]
    REF_GENERATED_TOKENS = kwargs["generated_tokens"]
    REF_LOGITS = kwargs["logits"]
    REF_LOSS = kwargs["loss"]

    # NOTE: we don't parallelize dropout layers
    # and activation functions
    {type(model.transformer.h[0].mlp.gelu_impl), type(model.transformer.h[0].self_attention.attention_dropout)}

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    parallelized_model = TensorParallel(model, parallel_context).parallelize()

    # NOTE: because pytorch also returns nested modules
    # and we only want to check the leaf modules,
    # so we filter out the nested modules
    # leaf_modules = get_leaf_modules(parallelized_model)
    # for module_name, module in leaf_modules:
    #     if type(module) in SKIP_MODULES:
    #         continue

    #     assert is_parallelized(module) is True, f"module {module_name} is not parallelized"

    generated_tokens = parallelized_model.generate(**input, **generation_configs)
    assert torch.allclose(generated_tokens, REF_GENERATED_TOKENS)

    outputs = parallelized_model(**input, labels=labels)
    # assert torch.allclose(p_output.logits, logits, rtol=1e-1)
    # assert torch.allclose(p_output.loss, loss, rtol=1e-1)
    # assert torch.allclose(outputs.logits, REF_LOGITS)
    # import torch

    def find_unequal_pos_idxs(outputs, REF_LOGITS):
        unequal_pos_idxs = []
        for pos_idx in range(REF_LOGITS.shape[1]):
            if not torch.allclose(outputs.logits[:, pos_idx], REF_LOGITS[:, pos_idx]):
                unequal_pos_idxs.append(pos_idx)
        return unequal_pos_idxs

    # for pos_idx in range(outputs.logits.shape[1]):
    #     assert torch.allclose(outputs.logits[:, pos_idx], REF_LOGITS[:, pos_idx]), f"pos_idx={pos_idx}"
    # idxs = find_unequal_pos_idxs(outputs, REF_LOGITS)
    assert torch.allclose(outputs.logits, REF_LOGITS)
    assert torch.allclose(outputs.loss, REF_LOSS)


@skip_in_github_actions
@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4])
def test_parallelize_a_transformer_and_inference(model, tokenizer, tensor_parallel_size):
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
    # labels = input["input_ids"]
    labels = torch.randint_like(inputs["input_ids"], low=100, high=200)

    generated_tokens = model.generate(**inputs, **GENERATION_CONFIGS)
    outputs = model(**inputs, labels=labels)

    # NOTE: we make a copy of the model before updating its weights
    # so the output of the model is not affected by the updated weights
    orig_model = deepcopy(model)
    loss = outputs.loss
    logits = outputs.logits

    kwargs = {
        "model": orig_model,
        "generation_configs": GENERATION_CONFIGS,
        "input": inputs,
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
    model = deepcopy(kwargs["model"])
    REF_MODEL = deepcopy(kwargs["updated_model"])

    lr = kwargs["lr"]
    input = kwargs["input"]
    labels = kwargs["labels"]
    kwargs["embedding_weight"]

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
    partitioned_embedding_weight = get_partition(
        REF_MODEL.transformer.word_embeddings.weight.data, dim=0, parallel_context=parallel_context
    )
    # # TODO: investigate why the rtol is so high
    # assert torch.allclose(p_embedding_weight, partitioned_embedding_weight, rtol=1e-1)
    if rank == 0:
        assert torch.allclose(p_embedding_weight, partitioned_embedding_weight)

    # for p, ref_p in zip(parallelized_model.parameters(), REF_MODEL.parameters()):
    #     # assert torch.allclose(p1, get_partition(p2, dim=0, parallel_context=parallel_context), rtol=1e-1)
    #     assert torch.allclose(p, get_partition(ref_p, dim=0, parallel_context=parallel_context))


@skip_in_github_actions
@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4])
def test_backward_pass_a_parallelized_transformers(model, tokenizer, tensor_parallel_size):
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    LR = 1e-3

    ORIG_MODEL = deepcopy(model)
    text = "Persistence is all you need."
    input = tokenizer(text, return_tensors="pt")
    labels = input["input_ids"]
    optim = SGD(model.parameters(), lr=LR)

    outputs = model(**input, labels=labels)

    # NOTE: we make a copy of the model before updating its weights
    # so the output of the model is not affected by the updated weights
    loss = outputs.loss

    optim.zero_grad()
    loss.backward()
    optim.step()

    kwargs = {
        "model": ORIG_MODEL,
        "lr": LR,
        "input": input,
        "labels": labels,
        # NOTE: this is the updated weight of the model
        "embedding_weight": model.transformer.word_embeddings.weight.data,
        "updated_model": deepcopy(model),
    }

    spawn(
        run_backward_a_parallelized_transformers,
        world_size=tensor_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        kwargs=kwargs,
    )
