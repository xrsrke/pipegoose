import pytest
import torch

from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel2._job.job import (
    ForwardJob,
    BackwardJob,
    JobStatus
)
from pipegoose.nn.pipeline_parallel2._package import Package
from pipegoose.nn.pipeline_parallel2._job.creator import create_job
from pipegoose.nn.pipeline_parallel2._job.job_type import JobType


@pytest.mark.parametrize("package, job_cls", [("forward_package", ForwardJob), ("backward_package", BackwardJob)])
def test_create_a_job_from_package(request, package, job_cls, parallel_context):
    LOGS = []

    def compute():
        LOGS.append(1)

    package = request.getfixturevalue(package)
    job = create_job(package, parallel_context)

    assert isinstance(job, job_cls)
    assert isinstance(job.key, str)
    assert job.status == JobStatus.PENDING


def test_the_job_status_after_executing_a_job(forward_job):
    # NOTE: this is directly execute the job instead of
    # waiting for a worker to pick up this job and execute it
    forward_job.compute()

    assert forward_job.status == JobStatus.EXECUTED


def test_the_output_package_after_executing_a_job(forward_job, parallel_context):
    PARALLEL_MODE = ParallelMode.PIPELINE
    ORIG_MICROBATCH_IDX = forward_job.package.metadata.microbatch_idx
    ORIG_PARTITION_IDX = forward_job.package.metadata.partition_idx
    SRC = parallel_context.get_global_rank()
    LOCAL_RANK = parallel_context.get_local_rank(PARALLEL_MODE)
    DST = parallel_context.get_next_local_rank(LOCAL_RANK, PARALLEL_MODE)

    output = forward_job.compute()

    assert forward_job.output == output
    assert isinstance(output, Package)

    assert isinstance(output.data, torch.Tensor)
    assert output.metadata.microbatch_idx == ORIG_MICROBATCH_IDX
    assert output.metadata.partition_idx == ORIG_PARTITION_IDX + 1

    assert output.metadata.job_type == JobType.FORWARD
    for key in vars(output.metadata.training).keys():
        # TODO: add test automatically switch to create new package
        # for different mix precision training
        assert getattr(output.metadata.training, key) == getattr(forward_job.package.metadata.training, key)

    # NOTE: we expect the metadata of the output package to
    # indicate which node executed it

    # TODO: update source rank and destination rank based on pipeline context
    # assert output.metadata.src == SRC
    # assert output.metadata.dst == DST
