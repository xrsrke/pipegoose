import torch


def get_memory_address_of_tensor_storage():
    pass


def mb_size_to_num_elements(mb: float, dtype: torch.dtype) -> int:
    """Convert a size in megabytes to a number of elements."""
    info_classes = {
        torch.float16: torch.finfo,
        torch.float32: torch.finfo,
        torch.float64: torch.finfo,
        torch.complex64: torch.finfo,
        torch.complex128: torch.finfo,
        torch.uint8: torch.iinfo,
        torch.int8: torch.iinfo,
        torch.int16: torch.iinfo,
        torch.int32: torch.iinfo,
        torch.int64: torch.iinfo,
    }

    if dtype not in info_classes:
        raise ValueError(f"Unsupported dtype: {dtype}.")

    bytes_per_dtype = info_classes[dtype](dtype).bits // 8
    bytes_per_mb = 1024 * 1024
    total_bytes = mb * bytes_per_mb
    return total_bytes // bytes_per_dtype
