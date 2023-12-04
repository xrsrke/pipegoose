import pytest
from transformers import AutoModelForCausalLM

from pipegoose.nn.expert_parallel.parallel_mapping import ExpertParallelMapping

MODEL_NAME = "Muennighoff/bloom-tiny-random"


@pytest.fixture(scope="session")
def model():
    return AutoModelForCausalLM.from_pretrained(MODEL_NAME)


@pytest.mark.skip(reason="Not implemented yet.")
def test_is_mlp_mapping(model):
    BLOOM_MLP_NAME = "transformer.h.{}.mlp"
    mappings = {}

    for name, _ in model.named_modules():
        # if "transformer.h.0.mlp" in name:
        #     assert 1 == 1

        mappings[name] = ExpertParallelMapping.is_mlp(name)

    for layer_idx in range(len(model.transformer.h)):
        assert mappings[BLOOM_MLP_NAME.format(layer_idx)] is True
