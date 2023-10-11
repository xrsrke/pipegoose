import torch


def get_tensor_storage_mem_loc(tensor: torch.Tensor) -> int:
    """Return the memory location of the tensor storage."""
    return tensor.storage().data_ptr()
