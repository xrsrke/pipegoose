import pytest
import torch
from torch import nn

from pipegoose.partitioning.profile import ProfileByMemory

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")


# @skip_if_no_cuda
@pytest.mark.skip("consider remove this module")
def test_profile_by_memory():
    # model = AutoModel.from_pretrained("gpt2")
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # text = [
    #     "Persistence is all you need.",
    #     "Persistence is all you need.",
    #     "Persistence is all you need."
    # ]
    # token_ids = tokenizer(text, return_tensors="pt")["input_ids"]

    model = nn.Sequential(*[nn.Linear(i + 1, i + 2) for i in range(6)])
    sample = torch.rand(7, 1)

    NUM_MODULES = sum(1 for _ in model.children())

    profiler = ProfileByMemory(model, torch.device("cpu"))
    sizes = profiler.profile(sample)

    assert len(sizes) == NUM_MODULES
    assert all(isinstance(size, int) for size in sizes)
