from copy import deepcopy

import pytest
import torch
from torch import nn

from pipegoose.nn.pipeline_parallel._job.creator import create_job
from pipegoose.nn.pipeline_parallel._job.job_type import JobType
from pipegoose.nn.pipeline_parallel._package import Metadata, Package, TrainingMetadata
from pipegoose.testing.utils import init_parallel_context

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


@pytest.fixture
def training_info():
    return {
        "n_microbatches": 5,
    }


@pytest.fixture(scope="session")
def parallel_context():
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1
    RANK = 0
    WORLD_SIZE = 1
    PORT = 12355

    parallel_context = init_parallel_context(
        rank=RANK,
        world_size=WORLD_SIZE,
        port=PORT,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
    )

    return parallel_context


@pytest.fixture(scope="session")
def pipeline_context(parallel_context):
    from pipegoose.nn.pipeline_parallel.pipeline_context import PipelineContext
    from pipegoose.nn.pipeline_parallel.scheduler import SchedulerType, get_scheduler

    # N_PARTITIONS = 3
    N_PARTITIONS = parallel_context.pipeline_parallel_size
    N_MICROBATCHES = 5

    scheduler = get_scheduler(SchedulerType.GPIPE)(N_MICROBATCHES, N_PARTITIONS)
    pipeline_context = PipelineContext(
        scheduler=scheduler,
        parallel_context=parallel_context,
    )

    return pipeline_context


@pytest.fixture
def base_package():
    MICROBATCH_IDX = 0
    PARTITION_IDX = 0
    IS_TRAINING = True
    IS_GRAD_ENABLED = True

    # NOTE: this is the package of an input microbatch
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


@pytest.fixture(scope="function")
def backward_package(base_package):
    from pipegoose.nn.pipeline_parallel.queue import (
        save_input_activations,
        save_output_activations,
    )

    backward_package = deepcopy(base_package)
    backward_package.metadata.src = 2
    backward_package.metadata.dst = 1
    backward_package.metadata.job_type = JobType.BACKWARD

    MICROBATCH_IDX = backward_package.metadata.microbatch_idx
    PARTITION_IDX = backward_package.metadata.partition_idx

    input = torch.randn(*INPUT_SHAPE, requires_grad=True)
    save_input_activations(input, MICROBATCH_IDX, PARTITION_IDX)

    linear = nn.Linear(*LINEAR_SHAPE)
    output = linear(input)
    INITIAL_GRADS = torch.ones_like(output)

    # NOTE: stores the output activations that the backward job
    # will use to compute the gradients
    save_output_activations(output, MICROBATCH_IDX, PARTITION_IDX)

    backward_package.data = torch.ones_like(INITIAL_GRADS)

    return backward_package


@pytest.fixture(scope="function")
def backward_job(backward_package, parallel_context, pipeline_context):
    def function():
        def backward_function(*args, **kwargs):
            return torch.randn(1)

        return backward_function

    job = create_job(function, backward_package, parallel_context, pipeline_context)
    return job


@pytest.fixture
def forward_function():
    return nn.Linear(*LINEAR_SHAPE)


@pytest.fixture(scope="function")
def forward_job(forward_package, forward_function):
    from pipegoose.nn.pipeline_parallel._job.forward import ForwardJob

    # return create_job(function, forward_package, parallel_context, pipeline_context)
    return ForwardJob(forward_function, forward_package)
