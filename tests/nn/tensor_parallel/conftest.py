import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "bigscience/bloom-560m"


@pytest.fixture(scope="session")
def model():
    return AutoModelForCausalLM.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)
