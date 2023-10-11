import pytest
import torch

from pipegoose.core.bucket.bucket import Bucket
from pipegoose.core.bucket.exception import BucketClosedError, BucketFullError


def test_add_a_tensor_to_bucket():
    BUCKET_SIZE = 1024
    DTYPE = torch.float32

    tensor = torch.randn(2, 4, dtype=DTYPE)
    TENSOR_STORAGE = tensor.storage()

    bucket = Bucket(BUCKET_SIZE, DTYPE)

    assert bucket.size == BUCKET_SIZE
    assert bucket.dtype == DTYPE
    assert bucket.available_size == BUCKET_SIZE
    assert len(bucket) == 0
    assert bucket.is_full is False

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


def test_add_tensor_that_larger_than_bucket_size():
    BUCKET_SIZE = 1024
    DTYPE = torch.float32
    tensor = torch.randn(2, BUCKET_SIZE, dtype=DTYPE)

    bucket = Bucket(BUCKET_SIZE, DTYPE)

    with pytest.raises(Exception):
        bucket.add_tensor(tensor)


def test_add_tensor_that_larger_than_available_space():
    BUCKET_SIZE = 1024
    DTYPE = torch.float32
    tensor = torch.randn(BUCKET_SIZE - 1)
    redundant_tensor = torch.randn(BUCKET_SIZE, dtype=DTYPE)

    bucket = Bucket(BUCKET_SIZE, DTYPE)

    bucket.add_tensor(tensor)

    with pytest.raises(BucketFullError):
        bucket.add_tensor(redundant_tensor)


def test_add_a_tensor_to_a_closed_bucket():
    BUCKET_SIZE = 1024
    DTYPE = torch.float32
    tensor = torch.randn(100)

    bucket = Bucket(BUCKET_SIZE, DTYPE)
    assert bucket.is_closed is False

    bucket.close()

    with pytest.raises(BucketClosedError):
        bucket.add_tensor(tensor)

    assert bucket.is_closed is True


def test_add_a_tensor_with_different_dtype_to_a_bucket():
    BUCKET_SIZE = 1024
    DTYPE = torch.float32
    tensor = torch.randn(10, dtype=torch.float16)

    bucket = Bucket(BUCKET_SIZE, DTYPE)

    with pytest.raises(Exception):
        bucket.add_tensor(tensor)


def test_flush_all_tensors_in_bucket():
    BUCKET_SIZE = 1024
    DTYPE = torch.float32
    x1 = torch.randn(10, dtype=DTYPE)
    x2 = torch.randn(20, dtype=DTYPE)

    bucket = Bucket(BUCKET_SIZE, DTYPE)
    bucket.add_tensor(x1)
    bucket.add_tensor(x2)
    bucket.clear()

    assert bucket.available_size == BUCKET_SIZE
    assert len(bucket) == 0
    # NOTE: how to test whether the bucket storage is deleted?
    # assert get_tensor_storage_mem_loc(x1) != bucket.storage().data_ptr()
    # assert get_tensor_storage_mem_loc(x2) != bucket.storage().data_ptr()


def test_delete_bucket_memory_storage():
    pass
