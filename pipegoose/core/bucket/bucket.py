import torch

from pipegoose.core.bucket.exception import BucketClosedError, BucketFullError
from pipegoose.distributed.parallel_context import ParallelContext


class Bucket:
    """Store tensors in a contiguous memory space."""

    def __init__(self, size: int, dtype: torch.dtype, parallel_context: ParallelContext):
        assert size > 0, "Bucket size must be greater than 0."
        # assert parallel_context is not None, "Parallel context must not be None."

        self.size = size
        self.dtype = dtype
        self.parallel_context = parallel_context

        self._buffer = torch.zeros(size, dtype=dtype)
        self._offset = 0
        self._is_closed = False
        self._num_tensors = 0

    @property
    def is_full(self) -> bool:
        return self._buffer.storage().size() == self._offset

    @property
    def available_size(self) -> int:
        return self._buffer.storage().size() - self._offset

    def add_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor), "Input must be a tensor."
        assert tensor.dtype == self._buffer.dtype, "Input tensor must have the same dtype as the bucket."

        if self.is_closed is True:
            raise BucketClosedError("Bucket is closed.")

        if self.is_full is True:
            raise BucketFullError("Bucket is full.")

        numel = tensor.numel()
        if numel > self.available_size:
            raise BucketFullError("Bucket does not have enough space.")

        self._buffer[self._offset : self._offset + numel].copy_(tensor.flatten())
        # NOTE: set the tensor's storage to its corresponding location in the bucket
        tensor.data = self._buffer[self._offset : self._offset + numel].view_as(tensor)
        self._offset += numel
        self._num_tensors += 1

        return tensor

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    def storage(self) -> torch.Storage:
        return self._buffer.storage()

    def close(self):
        """Close the bucket, and not allow any more tensors to be added to it."""
        assert self.is_closed is False, "Bucket is already closed."
        self._is_closed = True

    def free(self):
        """Delete all data in the bucket."""
        assert self._offset != 0, "Bucket is empty, so no need to free memory."

    def __len__(self) -> int:
        """Return the number of tensors in the bucket."""
        return self._num_tensors
