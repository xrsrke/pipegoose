import pytest
from torch import nn

from pipegoose.nn.pipeline_parallel2._job.backward import BackwardJob
from pipegoose.nn.pipeline_parallel2._job.creator import create_job
from pipegoose.nn.pipeline_parallel2._job.forward import ForwardJob
from pipegoose.nn.pipeline_parallel2._job.job import JobStatus

# NOTE: use for creating a forward job
function = nn.Linear(2, 4)


@pytest.mark.parametrize("package", ["forward_package", "backward_package"])
def test_the_job_status_after_executing_a_job(request, package, pipeline_context):
    package = request.getfixturevalue(package)
    job = create_job(function, package, pipeline_context)

    job.compute()

    assert job.status == JobStatus.EXECUTED


@pytest.mark.parametrize("package, job_cls", [("forward_package", ForwardJob), ("backward_package", BackwardJob)])
def test_create_a_job_from_package(request, package, forward_job, job_cls, pipeline_context):
    package = request.getfixturevalue(package)

    forward_job.compute()
    job = create_job(function, package, pipeline_context)

    assert isinstance(job, job_cls)
    assert isinstance(job.key, str)
    assert callable(job.function) is True
    assert job.status == JobStatus.PENDING
