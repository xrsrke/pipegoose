import pytest
from transformers import AutoModel

MODEL_NAME = "bigscience/bloom-560m"


@pytest.fixture(scope="session")
def model():
    return AutoModel.from_pretrained(MODEL_NAME)
