import pytest
import torch.distributed as dist
from torch import nn

from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel2._job.backward import BackwardJob
from pipegoose.nn.pipeline_parallel2._job.creator import create_job
from pipegoose.nn.pipeline_parallel2._job.forward import ForwardJob
from pipegoose.nn.pipeline_parallel2._job.job import JobStatus
from pipegoose.nn.pipeline_parallel2.sync.handshake import ProgressTracker
from pipegoose.nn.pipeline_parallel2.sync.progress_tracker import (
    get_progresses_from_pipeline_context,
)
from pipegoose.testing.utils import init_pipeline_context, spawn

# NOTE: use for creating a forward job
function = nn.Linear(2, 4)


# @pytest.mark.parametrize("package", ["forward_package", "backward_package"])
@pytest.mark.skip
def test_backward_job(backward_package, parallel_context, pipeline_context):
    # package = request.getfixturevalue(package)
    job = create_job(function, backward_package, parallel_context, pipeline_context)

    job.compute()

    assert job.status == JobStatus.EXECUTED


def run_create_a_job_from_package(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, package, job_cls
):
    MASTER_RANK = 0
    pipeline_context, parallel_context = init_pipeline_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    pipeline_context.forward()
    tracker = ProgressTracker(MASTER_RANK, parallel_context=parallel_context, parallel_mode=ParallelMode.GLOBAL)
    progresses = get_progresses_from_pipeline_context(pipeline_context)
    tracker.initiate(progresses)

    dist.barrier()

    job = create_job(function, package, parallel_context, pipeline_context)

    assert isinstance(job, job_cls)
    assert isinstance(job.key, str)
    assert callable(job.function) is True
    assert job.status == JobStatus.PENDING

    job.compute()

    assert job.status == JobStatus.EXECUTED


@pytest.mark.skip
@pytest.mark.parametrize("package, job_cls", [("forward_package", ForwardJob), ("backward_package", BackwardJob)])
def test_create_a_job_from_package(request, package, job_cls):
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 2
    DATA_PARALLEL_SIZE = 1
    WORLD_SIZE = TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE * DATA_PARALLEL_SIZE

    package = request.getfixturevalue(package)

    spawn(
        run_create_a_job_from_package,
        world_size=WORLD_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        package=package,
        job_cls=job_cls,
    )
