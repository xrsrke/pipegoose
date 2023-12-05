import random

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoTokenizer, BloomConfig, BloomForCausalLM

from pipegoose.distributed.functional import all_gather
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn import ExpertParallel
from pipegoose.nn.data_parallel.data_parallel import DataParallel
from pipegoose.nn.expert_parallel.loss import ExpertLoss
from pipegoose.nn.expert_parallel.routers import SwitchNoisePolicy, Top1Router
from pipegoose.testing.utils import get_microbatch, init_parallel_context, spawn

MODEL_NAME = "bigscience/bloom-560m"


@pytest.fixture
def model():
    config = BloomConfig(n_layer=4)
    model = BloomForCausalLM(config)
    return model


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def run_expert_parallel_with_data_parallel(
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
    router = kwargs["router"]
    NUM_EXPERTS = kwargs["num_experts"]

    # TODO: remove after adding seed to parallel_context
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    parallel_context = init_parallel_context(
        rank,
        world_size,
        port,
        tensor_parallel_size,
        pipeline_parallel_size,
        data_parallel_size,
    )
    # NOTE: each model replicas only train on a subset of data
    input_ids, attention_mask, labels = get_microbatch(
        kwargs["input"], kwargs["labels"], parallel_context, ParallelMode.EXPERT
    )
    loss_func = ExpertLoss(nn.CrossEntropyLoss())

    model = ExpertParallel(model, NUM_EXPERTS, mapping=mapping, router=router, parallel_context=parallel_context).parallelize()
    model = DataParallel(model, parallel_context).parallelize()
    optim = Adam(model.parameters(), lr=1e-3)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits[..., :-1, :].view(-1, outputs.logits.shape[-1])
    labels = labels[..., 1:].view(-1).to(logits.device)
    loss = loss_func(logits, labels)

    optim.zero_grad()
    loss.backward()

    expert_grad = list(model.transformer.h[0].mlp.parameters())[0]
    expert_grads = all_gather(expert_grad, parallel_context=parallel_context, parallel_mode=ParallelMode.EXPERT)
    expert_grads = torch.chunk(expert_grads, chunks=data_parallel_size, dim=0)

    # NOTE: check if expert grads are the same across data parallel dimension
    assert torch.allclose(*expert_grads)

    optim.step()


def test_expert_parallel_with_data_parallel(model, tokenizer):
    TENSOR_PARALLEL_SIZE = 2
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 2
    WORLD_SIZE = TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE * DATA_PARALLEL_SIZE

    NUM_EXPERTS = 2
    NUM_EXPERT_LAYERS = 2
    NUM_LAYERS = model.config.num_hidden_layers
    D_MODEL = model.config.hidden_size

    mapping = [layer_idx for layer_idx in random.sample(range(NUM_LAYERS - 1), NUM_EXPERT_LAYERS)]
    noise_policy = SwitchNoisePolicy()
    router = Top1Router(noise_policy, NUM_EXPERTS, D_MODEL)

    text = ["Persistence is all you need.", "Attention is all you need."]
    input = tokenizer(text, return_tensors="pt", padding=True)

    kwargs = {
        "input": input,
        "labels": input["input_ids"],
        "model": model,
        "mapping": mapping,
        "num_experts": NUM_EXPERTS,
        "router": router,
    }

    spawn(
        run_expert_parallel_with_data_parallel,
        world_size=WORLD_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        kwargs=kwargs,
    )
