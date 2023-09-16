import pytest
from transformers import AutoModelForCausalLM

MODEL_NAME = "Muennighoff/bloom-tiny-random"


@pytest.fixture(scope="session")
def model():
    return AutoModelForCausalLM.from_pretrained(MODEL_NAME)
