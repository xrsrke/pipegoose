from queue import Queue
from collections import OrderedDict

import pytest
import torch
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn.pipeline_parallel2._package import Package, Metadata, TrainingMetadata
from pipegoose.nn.pipeline_parallel2._job.job import (
    ForwardJob,
    BackwardJob,
    JobStatus
)
from pipegoose.nn.pipeline_parallel2._job.job_type import JobType
from pipegoose.nn.pipeline_parallel2._job.creator import JobCreator
from pipegoose.nn.pipeline_parallel2._job.register import JobRegister


@pytest.fixture(scope="session")
def parallel_context():
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1
    SEED = 69
    RANK = 0
    WORLD_SIZE = 1
    HOST = "localhost"
    PORT = 12355

    parallel_context = ParallelContext(
        rank=RANK,
        local_rank=RANK,
        world_size=WORLD_SIZE,
        local_world_size=WORLD_SIZE,
        host=HOST,
        port=PORT,
        backend="gloo",
        seed=SEED,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
    )

    return parallel_context


@pytest.fixture
def base_package():
    MICROBATCH_IDX = 1
    PARTITION_IDX = 2
    IS_TRAINING = True
    IS_GRAD_ENABLED = False

    SRC = 0
    DST = 1

    data = torch.randn(2, 4)

    metadata = Metadata(
        microbatch_idx=MICROBATCH_IDX,
        partition_idx=PARTITION_IDX,
        job_type=JobType.FORWARD,
        training=TrainingMetadata(
            is_training=IS_TRAINING,
            is_grad_enabled=IS_GRAD_ENABLED,
        ),
        src=SRC,
        dst=DST
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


@pytest.fixture
def forward_job(forward_package, parallel_context):
    LOGS = []

    def compute():
        LOGS.append(1)

    job = JobCreator(parallel_context).create(forward_package)

    return job


@pytest.fixture
def model():
    model = nn.Sequential(OrderedDict([
        ('layer1', nn.Sequential(OrderedDict([
            ('fc', nn.Linear(4, 8)),
            ('relu', nn.ReLU())
        ]))),
        ('layer2', nn.Sequential(OrderedDict([
            ('fc', nn.Linear(8, 4)),
            ('relu', nn.ReLU())
        ]))),
        ('layer3', nn.Sequential(OrderedDict([
            ('fc', nn.Linear(4, 8)),
            ('relu', nn.ReLU())
        ]))),
    ]))
    return model


# @pytest.mark.parametrize("get_package, job_cls", [(lambda forward_package: forward_package, ForwardJob), (backward_package, BackwardJob)])
def test_create_a_job(forward_package, parallel_context):
    LOGS = []

    def compute():
        LOGS.append(1)

    job_creator = JobCreator(parallel_context)

    job = job_creator.create(forward_package)

    assert isinstance(job, ForwardJob)
    assert isinstance(job.key, str)
    assert job.status == JobStatus.PENDING

    # NOTE: this is directly execute the job
    # instead of waiting for a worker to pick up this job
    # and execute it
    output = job.compute()
    assert isinstance(output, torch.Tensor)
    assert job.status == JobStatus.EXECUTED


def test_register_job(forward_job, parallel_context):
    JOB_QUEUE = Queue()

    job_register = JobRegister(queue=JOB_QUEUE, parallel_context=parallel_context)
    job_register.registry(forward_job)

    assert JOB_QUEUE.qsize() == 1
