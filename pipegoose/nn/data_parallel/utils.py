import torch


def free_storage(data: torch.Tensor):
    """Directly free the storage of a tensor.

    Args:
        data (torch.Tensor): a tensor
    """
    if data.storage().size() > 0:
        assert data.storage_offset() == 0
        data.storage().resize_(0)
