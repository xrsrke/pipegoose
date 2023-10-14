import pytest
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipegoose.nn.data_parallel.data_parallel import DataParallel
from pipegoose.optim.zero.optim import DistributedOptimizer
from pipegoose.testing.utils import init_parallel_context, spawn

MODEL_NAME = "prajjwal1/bert-tiny"


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
def test_hybrid_parallelism(tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    WORLD_SIZE = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
    GENERATION_CONFIGS = {"max_new_tokens": 1}

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    text = "Persistence is all you need."
    input = tokenizer(text, return_tensors="pt")
    labels = input["input_ids"]

    kwargs = {
        "model": model,
        "generation_configs": GENERATION_CONFIGS,
        "input": input,
        "labels": labels,
    }

    spawn(
        run_hybrid_parallelism,
        world_size=WORLD_SIZE,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        kwargs=kwargs,
    )
