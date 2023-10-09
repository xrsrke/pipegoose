import pytest
import torch
import torch.distributed as dist

from pipegoose.core.bucket.dist import BucketDistributor
from pipegoose.core.bucket.utils import mb_size_to_num_elements
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.testing.utils import init_parallel_context, spawn


def run_execute_a_tensor_that_larger_than_bucket_size(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
):
    # NOTE: append a tensor that is larger than the bucket size
    # and then execute the operation immediately
    PARALLEL_MODE = ParallelMode.DATA
    DTYPE = torch.float32
    BUCKET_SIZE_MB = 0.001
    NUM_ELEMNETS_IN_BUCKET = mb_size_to_num_elements(BUCKET_SIZE_MB, DTYPE)
    EXPECTED_OUTPUT = torch.arange(NUM_ELEMNETS_IN_BUCKET * 2).sum() * data_parallel_size

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    tensor = torch.arange(2 * NUM_ELEMNETS_IN_BUCKET, dtype=DTYPE)
    bucket_distributor = BucketDistributor(dist.all_reduce, BUCKET_SIZE_MB, parallel_context)
    bucket_distributor.execute(tensor, PARALLEL_MODE)

    output = tensor.sum()
    assert torch.equal(output, EXPECTED_OUTPUT)


@pytest.mark.parametrize("data_parallel_size", [1, 2])
def test_execute_a_tensor_that_larger_than_bucket_size(data_parallel_size):
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 1
    WORLD_SIZE = TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE * data_parallel_size

    spawn(
        run_execute_a_tensor_that_larger_than_bucket_size,
        world_size=WORLD_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=data_parallel_size,
    )
