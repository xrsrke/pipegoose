import pytest
import torch

from pipegoose.core.bucket.utils import mb_size_to_num_elements


@pytest.mark.parametrize(
    "mb_size, dtype, expected_num_elements",
    [
        (10, torch.int8, 10485760),
        (20, torch.float32, 5242880),
        (40, torch.float16, 20971520),
    ],
)
def test_mb_size_to_num_elements(mb_size, dtype, expected_num_elements):
    num_elements = mb_size_to_num_elements(mb_size, dtype)
    assert num_elements == expected_num_elements
