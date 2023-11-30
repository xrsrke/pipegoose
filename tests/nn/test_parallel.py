import pytest
from unittest.mock import MagicMock

import torch
from torch import nn
from transformers import BloomConfig, BloomForCausalLM
from pipegoose.nn.fusion import FusedBiasDropout, FusedBiasGelu
from pipegoose.nn.parallel import Parallel

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
NESTED_MODEL = nn.Sequential(
    nn.Linear(10, 10),
    nn.Sequential(
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(10, 10),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(10, 10),
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
    base_model_parallel = Parallel(module=NESTED_MODEL, parallel_context=MagicMock())
    fused_base_model = base_model_parallel.fuse([FusedBiasGelu, FusedBiasDropout])

    nested_model_parallel = Parallel(module=NESTED_MODEL, parallel_context=MagicMock())
    fused_nested_model = nested_model_parallel.fuse([FusedBiasGelu, FusedBiasDropout])
    
    bloom_560m_parallel = Parallel(module=BLOOM_560M, parallel_context=MagicMock())
    fused_bloom_module = bloom_560m_parallel.fuse([FusedBiasGelu, FusedBiasDropout])
    # TODO: To test, make sure both models' forward passes return the same results within a tolerance
    # Also, make sure that things *actually* got fused somehow, because the above condition would allow fuse()
    # to just be an identity function 

if __name__ == "__main__":
    test_parallel_fuse_with_gelu_dropout()
    