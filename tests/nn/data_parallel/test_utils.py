import pytest
import torch

from pipegoose.nn.data_parallel.utils import free_storage


def test_free_storage_occupied_by_one_tensor():
    x = torch.tensor([1, 2, 3, 4, 5])
    assert x.storage().size() == 5

    free_storage(x)

    assert x.storage().size() == 0
    assert x.storage().data_ptr() == 0


def test_free_storage_occupied_by_multiple_tensors():
    x = torch.tensor([1, 2, 3, 4, 5])
    y = x[1:]  # y shares the same storage with x but with different view

    with pytest.raises(AssertionError):
        free_storage(y)

    assert x.storage().size() == 5
    assert x.storage().data_ptr() != 0
