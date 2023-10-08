import pytest
import torch

from pipegoose.core.bucket.bucket import Bucket
from pipegoose.core.bucket.exception import BucketClosedError, BucketFullError


class FakeParallelContext:
    pass


def test_bucket():
    BUCKET_SIZE = 1024
    DTYPE = torch.float32

    tensor = torch.randn(2, 4, dtype=DTYPE)
    TENSOR_STORAGE = tensor.storage()

    parallel_context = FakeParallelContext()
    bucket = Bucket(BUCKET_SIZE, DTYPE, parallel_context)

    assert bucket.size == BUCKET_SIZE
    assert bucket.dtype == DTYPE
    assert bucket.available_size == BUCKET_SIZE
    assert len(bucket) == 0
    assert bucket.is_full is False
    assert bucket.is_closed is False

    new_tensor = bucket.add_tensor(tensor)

    assert isinstance(new_tensor, torch.Tensor)
    assert torch.equal(new_tensor, tensor)
    assert bucket.available_size == BUCKET_SIZE - new_tensor.numel()
    assert len(bucket) == 1
    # NOTE: the new tensor should be stored in the same storage as the bucket
    assert new_tensor.storage().data_ptr() == bucket.storage().data_ptr()
    # NOTE: the new tensor should have a different storage from the original tensor
    # since it's stored in the bucket
    assert new_tensor.storage().data_ptr() != TENSOR_STORAGE.data_ptr()

    # bucket.clear()

    # assert bucket.available_size == BUCKET_SIZE
    # assert len(bucket) == 0

    bucket.close()

    assert bucket.is_closed is True


def test_add_tensor_that_larger_than_bucket_size():
    BUCKET_SIZE = 1024
    DTYPE = torch.float32
    tensor = torch.randn(2, BUCKET_SIZE, dtype=DTYPE)

    parallel_context = FakeParallelContext()
    bucket = Bucket(BUCKET_SIZE, DTYPE, parallel_context)

    with pytest.raises(Exception):
        bucket.add_tensor(tensor)


def test_add_tensor_that_larger_than_available_space():
    BUCKET_SIZE = 1024
    DTYPE = torch.float32
    tensor = torch.randn(BUCKET_SIZE - 1)
    redundant_tensor = torch.randn(BUCKET_SIZE, dtype=DTYPE)

    parallel_context = FakeParallelContext()
    bucket = Bucket(BUCKET_SIZE, DTYPE, parallel_context)

    bucket.add_tensor(tensor)

    with pytest.raises(BucketFullError):
        bucket.add_tensor(redundant_tensor)


def test_add_a_tensor_to_a_closed_bucket():
    BUCKET_SIZE = 1024
    DTYPE = torch.float32
    tensor = torch.randn(BUCKET_SIZE - 1)

    parallel_context = FakeParallelContext()
    bucket = Bucket(BUCKET_SIZE, DTYPE, parallel_context)

    bucket.close()

    with pytest.raises(BucketClosedError):
        bucket.add_tensor(tensor)


def test_add_a_tensor_with_different_dtype_to_a_bucket():
    BUCKET_SIZE = 1024
    DTYPE = torch.float32
    tensor = torch.randn(10, dtype=torch.float16)

    parallel_context = FakeParallelContext()
    bucket = Bucket(BUCKET_SIZE, DTYPE, parallel_context)

    bucket.close()

    with pytest.raises(Exception):
        bucket.add_tensor(tensor)
