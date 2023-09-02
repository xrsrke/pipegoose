import pytest
from transformers import AutoModelForCausalLM

MODEL_NAME = "bigscience/bloom-560m"


@pytest.fixture(scope="session")
def model():
    return AutoModelForCausalLM.from_pretrained(MODEL_NAME)
