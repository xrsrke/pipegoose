import pytest
from copy import deepcopy
from unittest.mock import MagicMock

import torch
from torch import nn
from transformers import BloomConfig, BloomForCausalLM
from transformers.models.bloom.modeling_bloom import BloomGelu

from pipegoose.nn.fusion import FusedDropout, FusedGelu
from pipegoose.nn.parallel import Parallel

from torch.nn import GELU, Dropout, Module
# Construct a very basic model that inherits nn.Module, using GeLU and Dropout
BASE_MODEL = nn.Sequential(
    nn.Linear(10, 10),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(10, 10),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(10, 10)
)
from torch.fx import replace_pattern

# replace_pattern(torch.fx.symbolic_trace(BASE_MODEL), GELU, FusedGelu)

NESTED_MODEL = nn.Sequential(
    nn.Linear(10, 10),
    nn.Sequential(
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(10, 10),
        nn.GELU(),
        nn.Dropout(0.1),
        BASE_MODEL
    ),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(10, 10),
    nn.GELU(),
    nn.Linear(10, 10)
)
BLOOM_560M = BloomForCausalLM(BloomConfig())



def test_parallel_fuse_with_gelu_dropout():
    
    
    # fused_bloom_module2 = Parallel(module=deepcopy(BLOOM_560M), parallel_context=MagicMock()).fuse_md([FusedGelu, FusedDropout])
    # fused_bloom_module = Parallel(module=deepcopy(BLOOM_560M), parallel_context=MagicMock()).fuse([FusedGelu, FusedDropout])
    # fused_base_model = Parallel(module=deepcopy(BASE_MODEL), parallel_context=MagicMock()).fuse([FusedGelu, FusedDropout])
    # NOTE: This fails because using torch.fx cannot wrap builtin functions such as __len__, which is used by built-in Bloom 
    fused_nested_model = Parallel(module=deepcopy(BLOOM_560M), parallel_context=MagicMock()).fuse_fx([FusedGelu, FusedDropout])
    # NOTE: This fails because our manual fusion method cant handle nested models (e.g. multiple sequentials within each other)
    fused_nested_model = Parallel(module=deepcopy(NESTED_MODEL), parallel_context=MagicMock()).fuse([FusedGelu, FusedDropout])
    
    # For each model, make sure that no GeLU or Dropout layers remain
    for fused_model in [fused_nested_model]:
        for module in fused_model.modules():
            assert type(module) not in {nn.GELU, nn.Dropout, BloomGelu}


def test_parallel_fuse_with_gelu_dropout_train():
    # Generate some random data to train on
    batch_size = 16
    datapoint_count = 200
    dataset = [torch.randn(batch_size, 10) for _ in range(datapoint_count)]
    labels = [torch.randint_like(dataset[0], low=0, high=10) for _ in range(datapoint_count)]
    expected_outputs = [BASE_MODEL(batch) for batch in dataset]


    fused_nested_model = Parallel(module=deepcopy(BASE_MODEL), parallel_context=MagicMock()).fuse([FusedGelu, FusedDropout])
    actual_outputs = [fused_nested_model(batch) for batch in dataset]
    assert torch.allclose(expected_outputs, actual_outputs)

    loss_fn = nn.CrossEntropyLoss()
    for batch, label in zip(dataset, labels):
        nested_ouptut = BASE_MODEL(batch)
        fused_outputs = fused_nested_model(batch)

        nested_loss = loss_fn(nested_ouptut, label)
        fused_loss = loss_fn(fused_outputs, label)

        nested_loss.backward()
        fused_loss.backward()

    # Re-run both trained models
    for batch in dataset:
        assert torch.allclose(BASE_MODEL(batch), fused_nested_model(batch))






if __name__ == "__main__":
    test_parallel_fuse_with_gelu_dropout()
    test_parallel_fuse_with_gelu_dropout_train()
    