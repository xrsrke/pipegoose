import pytest
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipegoose.nn import DataParallel
from pipegoose.optim.zero.optim import DistributedOptimizer
from pipegoose.testing.utils import init_parallel_context, spawn

MODEL_NAME = "prajjwal1/bert-tiny"


@pytest.fixture(scope="module")
def model():
    return AutoModelForCausalLM.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def run_hybrid_parallelism(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, kwargs):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    parallelized_model = DataParallel(kwargs["model"], parallel_context).parallelize()
    optim = Adam(parallelized_model.parameters())
    dist_optim = DistributedOptimizer(optim, parallel_context)

    output = parallelized_model(**kwargs["input"], labels=kwargs["labels"])
    loss = output.loss

    dist_optim.zero_grad()
    loss.backward()
    dist_optim.step()


@pytest.mark.parametrize("tensor_parallel_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [2])
@pytest.mark.parametrize("data_parallel_size", [2])
def test_hybrid_parallelism(model, tokenizer, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    WORLD_SIZE = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
    GENERATION_CONFIGS = {"max_new_tokens": 1}

    text = "Persistence is all you need."
    input = tokenizer(text, return_tensors="pt")
    labels = input["input_ids"]

    kwargs = {
        "model": model,
        "generation_configs": GENERATION_CONFIGS,
        "input": input,
        "labels": labels,
        # "generated_tokens": generated_tokens.detach(),
        # "logits": logits.detach(),
        # "loss": loss.detach(),
    }

    spawn(
        run_hybrid_parallelism,
        world_size=WORLD_SIZE,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        kwargs=kwargs,
    )
