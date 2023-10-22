from copy import deepcopy

import pytest
import torch
from torch.optim import SGD
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.data_parallel.data_parallel import DataParallel
from pipegoose.testing.utils import (
    calculate_parameter_similarity,
    init_parallel_context,
    spawn,
)

MODEL_NAME = "prajjwal1/bert-tiny"


# @pytest.fixture(scope="module")
# def model():
#     return AutoModelForCausalLM.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def run_parallelize_a_transformers_and_inference(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, kwargs
):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    parallelized_model = DataParallel(kwargs["model"], parallel_context).parallelize()

    p_generated_tokens = parallelized_model.generate(**kwargs["input"], **kwargs["generation_configs"])
    assert torch.allclose(p_generated_tokens, kwargs["generated_tokens"])

    p_output = parallelized_model(**kwargs["input"], labels=kwargs["labels"])
    assert torch.allclose(p_output.logits, kwargs["logits"], rtol=1e-1)
    assert torch.allclose(p_output.loss, kwargs["loss"], rtol=1e-1)


@pytest.mark.parametrize("data_parallel_size", [1, 2])
def test_parallelize_a_transformer_and_inference(model, tokenizer, data_parallel_size):
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 1

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
        world_size=data_parallel_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=data_parallel_size,
        kwargs=kwargs,
    )


def run_backward_a_parallelized_transformers(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, kwargs
):
    def get_microbatch(inputs, labels):
        local_rank = parallel_context.get_local_rank(ParallelMode.DATA)
        input_chunks = torch.chunk(inputs["input_ids"], chunks=world_size, dim=0)
        attention_chunks = torch.chunk(inputs["attention_mask"], chunks=world_size, dim=0)
        label_chunks = torch.chunk(labels, chunks=world_size, dim=0)
        return input_chunks[local_rank], attention_chunks[local_rank], label_chunks[local_rank]

    model = deepcopy(kwargs["model"])
    UPDATED_MODEL = deepcopy(kwargs["updated_model"])
    LR = kwargs["lr"]
    inputs = kwargs["inputs"]
    labels = kwargs["labels"]

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    input_ids, attention_mask, labels = get_microbatch(inputs, labels)
    parallelized_model = DataParallel(model, parallel_context).parallelize()
    optim = SGD(parallelized_model.parameters(), lr=LR)

    optim.zero_grad()
    outputs = parallelized_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    loss = outputs.loss
    loss.backward()
    optim.step()

    # NOTE: after averaging the gradient, we expect the gradient of a replica
    # that trains on a subset of data to be equal to the gradient of
    # the original model that trains on the whole set of data
    for p, ref_p in zip(parallelized_model.parameters(), UPDATED_MODEL.parameters()):
        assert torch.allclose(p, ref_p)


@pytest.mark.parametrize("data_parallel_size", [1, 2])
def test_backward_pass_a_parallelized_transformers(tokenizer, data_parallel_size):
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 1

    # NOTE: if use small learning rate,
    # the updated model and the original model's weights can be identical in some cases
    # this could leads to wrong test
    LR = 1e-1

    text = ["Persistence is all you need.", "3D parallelism is all you need."]
    inputs = tokenizer(text, return_tensors="pt", padding="longest")
    labels = torch.randint_like(inputs["input_ids"], low=100, high=200)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    ORIG_MODEL = deepcopy(model)
    optim = SGD(model.parameters(), lr=LR)
    optim.zero_grad()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optim.step()

    # NOTE: if some cases, the updated model and the original model's weights can be identical
    # so we need to make sure the updated model and the original model's weights are different
    similarity = calculate_parameter_similarity(ORIG_MODEL, model)
    assert similarity < 0.95, f"Two models should be different before training. Similarity: {similarity}"

    kwargs = {
        "model": ORIG_MODEL,
        "updated_model": model,
        "lr": LR,
        "inputs": inputs,
        "labels": labels,
    }

    spawn(
        run_backward_a_parallelized_transformers,
        world_size=data_parallel_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=data_parallel_size,
        kwargs=kwargs,
    )
