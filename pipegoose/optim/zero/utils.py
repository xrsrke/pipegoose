from typing import List

import torch
from torch._utils import _flatten_dense_tensors


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
