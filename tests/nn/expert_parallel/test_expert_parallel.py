import random

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipegoose.nn import ExpertParallel
from pipegoose.nn.expert_parallel.layers import ExpertLayer
from pipegoose.testing.utils import init_parallel_context, spawn

MODEL_NAME = "bigscience/bloom-560m"


class DummyRouter:
    def __init__(self, num_experts):
        self.num_experts = num_experts

    def __call__(self, inputs):
        n_tokens = inputs.shape[0] * inputs.shape[1]
        return torch.randint(0, self.num_experts, (n_tokens,)), None, None


@pytest.fixture
def model():
    # config = BloomConfig()
    # model = BloomForCausalLM(config)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return model


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def run_expert_parallel(
    rank,
    world_size,
    port,
    tensor_parallel_size,
    pipeline_parallel_size,
    data_parallel_size,
    kwargs,
):
    model = kwargs["model"]
    mapping = kwargs["mapping"]
    NUM_EXPERTS = kwargs["num_experts"]
    router = kwargs["router"]

    parallel_context = init_parallel_context(
        rank,
        world_size,
        port,
        tensor_parallel_size,
        pipeline_parallel_size,
        data_parallel_size,
    )
    model = ExpertParallel(model, NUM_EXPERTS, mapping=mapping, router=router, parallel_context=parallel_context).parallelize()

    # NOTE: check the specified layers are replaced with expert layers
    assert [isinstance(model.transformer.h[i].mlp, ExpertLayer) for i in mapping].count(True) == len(mapping)

    # NOTE: we haven't go through any weight update yet
    # so the logits should be the same
    outputs = model(**kwargs["input"], labels=kwargs["labels"])
    assert outputs.logits.shape == kwargs["ref_logits"].shape


@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 8])
def test_expert_parallel(model, tokenizer, tensor_parallel_size):
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1
    WORLD_SIZE = tensor_parallel_size * PIPELINE_PARALLEL_SIZE * DATA_PARALLEL_SIZE

    NUM_LAYERS = model.config.num_hidden_layers
    NUM_EXPERTS = 8
    NUM_EXPERT_LAYERS = 2

    mapping = [layer_idx for layer_idx in random.sample(range(NUM_LAYERS - 1), NUM_EXPERT_LAYERS)]
    router = DummyRouter(NUM_EXPERTS)

    text = "Persistence is all you need."
    input = tokenizer(text, return_tensors="pt")
    outputs = model(**input, labels=input["input_ids"])

    kwargs = {
        "input": input,
        "labels": input["input_ids"],
        "model": model,
        "mapping": mapping,
        "num_experts": NUM_EXPERTS,
        "router": router,
        "ref_logits": outputs.logits.detach(),
    }

    spawn(
        run_expert_parallel,
        world_size=WORLD_SIZE,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        kwargs=kwargs,
    )
