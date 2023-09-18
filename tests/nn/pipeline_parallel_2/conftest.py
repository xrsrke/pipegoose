import pytest
import torch
from torch import nn

from pipegoose.nn.pipeline_parallel2._job.creator import create_job
from pipegoose.nn.pipeline_parallel2._job.job_type import JobType
from pipegoose.nn.pipeline_parallel2._package import Metadata, Package, TrainingMetadata
from pipegoose.testing.utils import init_pipeline_context

# NOTE: it should be compatible to perform
# matrix multiplication with the job's function
INPUT_SHAPE = (
    4,
    2,
)
LINEAR_SHAPE = (
    2,
    4,
)


@pytest.fixture(scope="session")
def pipeline_context():
    ENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1
    RANK = 0
    WORLD_SIZE = 1
    PORT = 12355

    N_PARTITIONS = 3
    N_MICROBATCHES = 5

    pipeline_context = init_pipeline_context(
        rank=RANK,
        world_size=WORLD_SIZE,
        port=PORT,
        tensor_parallel_size=ENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        n_partitions=N_PARTITIONS,
        n_microbatches=N_MICROBATCHES,
    )

    return pipeline_context


@pytest.fixture
def base_package():
    MICROBATCH_IDX = 0
    PARTITION_IDX = 0
    IS_TRAINING = True
    IS_GRAD_ENABLED = True

    SRC = 0
    DST = 1

    data = torch.randn(*INPUT_SHAPE, requires_grad=IS_GRAD_ENABLED)

    metadata = Metadata(
        microbatch_idx=MICROBATCH_IDX,
        partition_idx=PARTITION_IDX,
        job_type=JobType.FORWARD,
        training=TrainingMetadata(
            is_training=IS_TRAINING,
            is_grad_enabled=IS_GRAD_ENABLED,
        ),
        src=SRC,
        dst=DST,
    )

    return Package(data, metadata)


@pytest.fixture
def forward_package(base_package):
    # NOTE: package for forward job
    base_package.metadata.job_type = JobType.FORWARD
    return base_package


@pytest.fixture
def backward_package(base_package):
    # NOTE: package for backward job
    base_package.metadata.job_type = JobType.BACKWARD
    return base_package


@pytest.fixture(scope="function")
def backward_job(backward_package, pipeline_context):
    def function():
        def backward_function(*args, **kwargs):
            return torch.randn(1)

        return backward_function

    return create_job(function, backward_package, pipeline_context)


@pytest.fixture(scope="function")
def forward_job(forward_package, pipeline_context):
    function = nn.Linear(*LINEAR_SHAPE)
    return create_job(function, forward_package, pipeline_context)
