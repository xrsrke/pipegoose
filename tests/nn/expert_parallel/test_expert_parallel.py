import pytest
from torch import nn

from pipegoose.nn import ExpertParallel
from pipegoose.testing.utils import init_parallel_context, spawn


def run_expert_parallel(
    rank,
    world_size,
    port,
    tensor_parallel_size,
    pipeline_parallel_size,
    data_parallel_size,
    kwargs,
):
    parallel_context = init_parallel_context(
        rank,
        world_size,
        port,
        tensor_parallel_size,
        pipeline_parallel_size,
        data_parallel_size,
    )
    parallel_context = None

    NUM_EXPERTS = 8
    expert = nn.Sequential(
        nn.Linear(1024, 1024 * 4),
        nn.ReLU(),
        nn.Linear(1024 * 4, 1024),
    )

    ExpertParallel(model, NUM_EXPERTS, expert, parallel_context).parallelize()

    # NOTE: check if the model has expert layers, and the number of experts is correct
    # outputs = model(**kwargs["input"], labels=kwargs["labels"])


@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 8])
def test_parallelize_a_transformer_and_inference(model, tokenizer, tensor_parallel_size):
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1
    WORLD_SIZE = tensor_parallel_size * PIPELINE_PARALLEL_SIZE * DATA_PARALLEL_SIZE

    text = "Persistence is all you need."
    input = tokenizer(text, return_tensors="pt")
    kwargs = {"input": input, "labels": input["input_ids"]}

    spawn(
        run_expert_parallel,
        world_size=WORLD_SIZE,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        kwargs=kwargs,
    )
