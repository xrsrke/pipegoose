from itertools import chain

import torch

from pipegoose.optim.zero.utils import (
    copy_flatten_tensor_to_unflatten_tensors,
    flatten_a_list_tensor,
)


def test_flatten_a_list_tensor():
    tensor_list = [torch.rand(2, 3) for _ in range(5)]

    flat_tensor = flatten_a_list_tensor(tensor_list)

    assert flat_tensor.numel() == sum(t.numel() for t in tensor_list)
    assert flat_tensor.shape == (sum(t.numel() for t in tensor_list),)

    original_elements = torch.tensor(list(chain.from_iterable(t.tolist() for t in tensor_list)))
    assert torch.equal(original_elements.view(-1), flat_tensor)


def test_copy_flatten_tensor_to_unflatten_tensors():
    tensor_list = [torch.rand(2, 3) for _ in range(5)]
    flat_tensor = flatten_a_list_tensor(tensor_list)
    new_tensor_list = [torch.randn_like(t) for t in tensor_list]

    copy_flatten_tensor_to_unflatten_tensors(flat_tensor, new_tensor_list)

    for original, copied in zip(tensor_list, new_tensor_list):
        assert torch.equal(original, copied)
