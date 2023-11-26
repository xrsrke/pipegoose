import random
from functools import partial

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoTokenizer, BloomConfig, BloomForCausalLM

from pipegoose.nn import ExpertParallel
from pipegoose.nn.expert_parallel.layers import ExpertLayer
from pipegoose.nn.expert_parallel.utils import get_num_local_experts
from pipegoose.testing.utils import init_parallel_context, spawn
from pipegoose.nn.expert_parallel.routers import RouterOutput
from pipegoose.nn.expert_parallel.loss import ExpertLoss

MODEL_NAME = "bigscience/bloom-560m"


class DummyRouter:
    def __init__(self, num_experts):
        self.num_experts = num_experts

    def __call__(self, inputs):
        n_tokens = inputs.shape[0] * inputs.shape[1]
        return RouterOutput(
            torch.randint(0, self.num_experts, (n_tokens,)),
            None,
            torch.tensor(0.0),
            torch.tensor(0.0),
        )


@pytest.fixture
def model():
    config = BloomConfig(n_layer=4)
    model = BloomForCausalLM(config)
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
    router = kwargs["router"]
    REF_LOSS = kwargs["ref_loss"]
    REF_LOGITS = kwargs["ref_logits"]
    NUM_EXPERTS = kwargs["num_experts"]

    # TODO: remove after adding seed to parallel_context
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # NOTE: add hooks to all experts, and record which experts are routed to
    # then check if the gradients are flowing to those routed experts
    # and not flowing to other non-routed experts
    routed_experts = set()

    def get_layer_idx(name):
        return int(name.split(".")[2])

    def record_dispatch_experts(model):
        def log_routed_expert(module, grad_input, grad_output, key):
            if grad_output[0] is not None:
                routed_experts.add(key)

        for name, module in model.named_modules():
            if isinstance(module, ExpertLayer):
                layer_idx = get_layer_idx(name)
                for expert_idx, expert in enumerate(module.experts):
                    key = (layer_idx, expert_idx)
                    expert.register_backward_hook(partial(log_routed_expert, key=key))

    loss_func = ExpertLoss(nn.CrossEntropyLoss(), aux_weight=0.1, z_weight=0.1)

    parallel_context = init_parallel_context(
        rank,
        world_size,
        port,
        tensor_parallel_size,
        pipeline_parallel_size,
        data_parallel_size,
    )
    model = ExpertParallel(
        model,
        NUM_EXPERTS,
        mapping=mapping,
        router=router,
        parallel_context=parallel_context,
        expert_context=loss_func.expert_context
    ).parallelize()
    optim = Adam(model.parameters(), lr=1e-3)

    # NOTE: check the specified layers are replaced with expert layers
    assert [isinstance(model.transformer.h[i].mlp, ExpertLayer) for i in mapping].count(True) == len(mapping)

    # NOTE: count the number of expert layers
    EXPECTED_NUM_EXPERT_LAYERS = len(mapping)
    num_expert_layers = sum(isinstance(module, ExpertLayer) for _, module in model.named_modules())
    assert num_expert_layers == EXPECTED_NUM_EXPERT_LAYERS

    NUM_LOCAL_EXPERTS = get_num_local_experts(NUM_EXPERTS, parallel_context)
    # NOTE: count the number of experts per layer
    for name, module in model.named_modules():
        if isinstance(module, ExpertLayer):
            num_experts = len(module.experts)
            assert num_experts == NUM_LOCAL_EXPERTS

    record_dispatch_experts(model)

    # NOTE: test the parameters of the MoE model to be equal to
    # the original model without that parallelized expert layer,
    # and the parameters of that expert layer

    # NOTE: we haven't go through any weight update yet
    # so the logits should be the same
    outputs = model(**kwargs["input"])

    # compute the loss
    logits = outputs.logits[..., :-1, :].view(-1, outputs.logits.shape[-1])
    labels = kwargs["labels"][..., 1:].view(-1).to(logits.device)
    loss = loss_func(logits, labels)

    # assert torch.allclose(outputs.logits, REF_LOGITS)
    assert outputs.logits.shape == REF_LOGITS.shape
    assert torch.allclose(loss, REF_LOSS)

    optim.zero_grad()
    loss.backward()
    optim.step()

    # NOTE: After the backward pass, check if the gradients flowing to the routed experts
    # and not flowing to other non-routed experts
    for name, module in model.named_modules():
        if isinstance(module, ExpertLayer):
            layer_idx = get_layer_idx(name)
            for expert_idx, expert in enumerate(module.experts):
                if (layer_idx, expert_idx) in routed_experts:
                    assert all(p.grad is not None for p in expert.parameters())
                else:
                    assert all(p.grad is None for p in expert.parameters())


@pytest.mark.parametrize("tensor_parallel_size, num_experts", [(1, 1), (2, 2), (2, 4), (2, 8), (8, 8)])
def test_expert_parallel(model, tokenizer, tensor_parallel_size, num_experts):
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1
    WORLD_SIZE = tensor_parallel_size * PIPELINE_PARALLEL_SIZE * DATA_PARALLEL_SIZE

    NUM_LAYERS = model.config.num_hidden_layers
    NUM_EXPERT_LAYERS = 2

    mapping = [layer_idx for layer_idx in random.sample(range(NUM_LAYERS - 1), NUM_EXPERT_LAYERS)]
    router = DummyRouter(num_experts)

    text = "Persistence is all you need."
    input = tokenizer(text, return_tensors="pt")
    outputs = model(**input, labels=input["input_ids"])

    kwargs = {
        "input": input,
        "labels": input["input_ids"],
        "model": model,
        "mapping": mapping,
        "num_experts": num_experts,
        "router": router,
        "ref_logits": outputs.logits.detach(),
        "ref_loss": outputs.loss.detach(),
    }

    spawn(
        run_expert_parallel,
        world_size=WORLD_SIZE,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        kwargs=kwargs,
    )
