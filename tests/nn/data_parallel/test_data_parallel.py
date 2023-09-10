from copy import deepcopy

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.data_parallel.data_parallel import DataParallel
from pipegoose.testing.utils import spawn


MODEL_NAME = "bigscience/bloom-560m"


@pytest.fixture()
def model():
    return AutoModelForCausalLM.from_pretrained(MODEL_NAME)


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


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

    GENERATION_CONFIGS = {
        "max_new_tokens": 1
    }

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
        kwargs=kwargs
    )
