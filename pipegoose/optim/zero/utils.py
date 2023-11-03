from typing import List

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def delete_tensor_from_memory(tensor):
    """
    Delete a tensor from memory

    Args:
        tensor (torch.Tensor): the tensor to be deleted
    """
    del tensor
    torch.cuda.empty_cache()


def flatten_a_list_tensor(list: List[torch.Tensor]) -> torch.Tensor:
    """Flatten a list of tensors into a single tensor."""
    return _flatten_dense_tensors(list)


def copy_flatten_tensor_to_unflatten_tensors(flat: torch.Tensor, tensors: List[torch.Tensor]):
    """Copied the data in a flatten tensor to its original unflatten tensors."""
    for tensor, flat_data in zip(tensors, _unflatten_dense_tensors(flat, tensors)):
        tensor.data.copy_(flat_data)
