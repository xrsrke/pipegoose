from copy import deepcopy

import pytest
import torch
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipegoose.nn.tensor_parallel.tensor_parallel import TensorParallel
from pipegoose.testing.utils import (
    get_partition,
    init_parallel_context,
    skip_in_github_actions,
    spawn,
)

MODEL_NAME = "bigscience/bloom-560m"


def run_hybrid_parallelism(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, kwargs):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    model = kwargs["model"]
    model = TensorParallel(model, parallel_context).parallelize()
    # model = DataParallel(model, parallel_context).parallelize()

    optim = Adam(model.parameters())
    # dist_optim = DistributedOptimizer(optim, parallel_context)

    output = model(**kwargs["input"], labels=kwargs["labels"])
    loss = output.loss

    optim.zero_grad()
    loss.backward()
    optim.step()

    for p1, p2 in zip(model.parameters(), kwargs["updated_model"].parameters()):
        assert torch.allclose(p1, get_partition(p2, dim=0, parallel_context=parallel_context), rtol=1e-1)


@skip_in_github_actions
@pytest.mark.parametrize("tensor_parallel_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1])
@pytest.mark.parametrize("data_parallel_size", [1])
def test_hybrid_parallelism(tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    WORLD_SIZE = tensor_parallel_size * pipeline_parallel_size * data_parallel_size

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    optim = Adam(model.parameters())
    ORIG_MODEL = deepcopy(model)

    text = "Persistence is all you need."
    input = tokenizer(text, return_tensors="pt")
    labels = input["input_ids"]

    output = model(**input, labels=labels)
    loss = output.loss
    optim.zero_grad()
    loss.backward()
    optim.step()

    kwargs = {
        "model": ORIG_MODEL,
        "updated_model": model,
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
