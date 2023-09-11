import pytest
import torch

from pipegoose.nn.pipeline_parallel2._job.job import (
    ForwardJob,
    BackwardJob,
    JobStatus
)
from pipegoose.nn.pipeline_parallel2._job.creator import JobCreator


@pytest.mark.parametrize("package, job_cls", [("forward_package", ForwardJob), ("backward_package", BackwardJob)])
def test_create_a_job(request, package, job_cls, parallel_context):
    LOGS = []

    def compute():
        LOGS.append(1)

    package = request.getfixturevalue(package)
    job_creator = JobCreator(parallel_context)

    job = job_creator.create(package)

    assert isinstance(job, job_cls)
    assert isinstance(job.key, str)
    assert job.status == JobStatus.PENDING

    # NOTE: this is directly execute the job
    # instead of waiting for a worker to pick up this job
    # and execute it
    output = job.compute()
    assert isinstance(output, torch.Tensor)
    assert job.status == JobStatus.EXECUTED
