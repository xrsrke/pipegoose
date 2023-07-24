import torch


def _is_using_full_storage(input: torch.Tensor) -> bool:
    """Check if the tensor is using full storage.

    Args:
        tensor (torch.Tensor): a tensor

    Returns:
        bool: True if the tensor is using full storage, otherwise False
    """
    total_storage = input.storage().size()
    n_elements = input.numel()

    if total_storage == n_elements:
        return True
    else:
        return False


def free_storage(input: torch.Tensor) -> None:
    """Directly free the storage of a tensor.

    Args:
        input (torch.Tensor): a tensor
    """
    if input.storage().size() > 0:
        assert _is_using_full_storage(input) is True
        input.storage().resize_(0)
