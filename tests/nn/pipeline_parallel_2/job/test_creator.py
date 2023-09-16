import pytest
import torch
from torch import nn

from pipegoose.nn.pipeline_parallel2._job.creator import create_job
from pipegoose.nn.pipeline_parallel2._job.job import BackwardJob, ForwardJob, JobStatus
from pipegoose.nn.pipeline_parallel2._job.job_type import JobType
from pipegoose.nn.pipeline_parallel2._package import Package
from pipegoose.nn.pipeline_parallel2._utils import sleep
from pipegoose.nn.pipeline_parallel2.queue import JobQueue
from pipegoose.testing.utils import init_pipeline_context, spawn

# NOTE: use for creating a forward job
function = nn.Linear(2, 4)


def run_check_the_job_status_after_executing_a_job(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, package
):
    pipeline_context = init_pipeline_context(
        rank,
        world_size,
        port,
        tensor_parallel_size,
        pipeline_parallel_size,
        data_parallel_size,
    )
    job = create_job(function, package, pipeline_context)

    job.compute()

    assert job.status == JobStatus.EXECUTED


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
@pytest.mark.parametrize("package", ["forward_package", "backward_package"])
def test_the_job_status_after_executing_a_job(request, pipeline_parallel_size, package):
    TENSOR_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    package = request.getfixturevalue(package)

    spawn(
        run_check_the_job_status_after_executing_a_job,
        world_size=pipeline_parallel_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=DATA_PARALLEL_SIZE,
        package=package,
    )


def run_execute_a_forward_job(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, package
):
    # NOTE: (microbatch_idx, partition_idx) -> (microbatch_idx, next_partition_idx)
    OUTPUT_DESTINATION = {
        (0, 0): (0, 1),
        (0, 1): (0, 2),
        (1, 0): (1, 1),
        (1, 1): (1, 2),
        (2, 0): (2, 1),
        (2, 1): (2, 2),
        (3, 0): (3, 1),
        (3, 1): (3, 2),
        (4, 0): (4, 1),
        (4, 1): (4, 2),
    }

    pipeline_context = init_pipeline_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    forward_job = create_job(function, package, pipeline_context)

    ORIG_MICROBATCH_IDX = forward_job.input.metadata.microbatch_idx
    ORIG_PARTITION_IDX = forward_job.input.metadata.partition_idx

    output = forward_job.compute()

    assert forward_job.output == output
    assert isinstance(output, Package)

    assert isinstance(output.data, torch.Tensor)
    assert OUTPUT_DESTINATION[(ORIG_MICROBATCH_IDX, ORIG_PARTITION_IDX)] == (
        output.metadata.microbatch_idx,
        output.metadata.partition_idx,
    )

    assert output.metadata.job_type == JobType.FORWARD

    for key in vars(output.metadata.training).keys():
        # TODO: add test automatically switch to create new package
        # for different mix precision training
        assert getattr(output.metadata.training, key) == getattr(forward_job.input.metadata.training, key)

    # NOTE: we expect the metadata of the output package to
    # indicate which node executed it
    # TODO: update source rank and destination rank based on pipeline context
    assert isinstance(output.metadata.src, int)
    assert isinstance(output.metadata.dst, int)

    if world_size > 1:
        from pipegoose.nn.pipeline_parallel2._comm import RECV_QUEUE

        sleep(5)
        assert RECV_QUEUE.qsize() == 1

        received_package = RECV_QUEUE.get()
        assert isinstance(received_package, Package)
        assert received_package.metadata.dst == rank


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2, 5])
@pytest.mark.parametrize("package", ["forward_package"])
def test_execute_a_forward_job(request, pipeline_parallel_size, package):
    TENSOR_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    package = request.getfixturevalue(package)

    spawn(
        run_execute_a_forward_job,
        world_size=pipeline_parallel_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=DATA_PARALLEL_SIZE,
        package=package,
    )


def run_create_a_job_from_package(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, package, job_cls
):
    pipeline_context = init_pipeline_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    job = create_job(function, package, pipeline_context)

    assert isinstance(job, job_cls)
    assert isinstance(job.key, str)
    assert callable(job.function) is True
    assert job.status == JobStatus.PENDING


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
@pytest.mark.parametrize("package, job_cls", [("forward_package", ForwardJob), ("backward_package", BackwardJob)])
def test_create_a_job_from_package(request, pipeline_parallel_size, package, job_cls):
    TENSOR_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    package = request.getfixturevalue(package)

    spawn(
        run_create_a_job_from_package,
        world_size=pipeline_parallel_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=DATA_PARALLEL_SIZE,
        package=package,
        job_cls=job_cls,
    )


@pytest.mark.skip
def test_execute_a_forward_job_and_send_the_output(forward_job, parallel_context):
    pass


@pytest.mark.skip
def test_create_forward_job_that_schedule_a_backward_job(rank, forward_package):
    rank = None
    SRC = 1
    DST = 0

    if rank == SRC:
        input = torch.randn(4, 2, requires_grad=True)
        forward_job = create_job(function, forward_package)

        package = forward_job.compute(input)
        package.data.sum().backward()

    elif rank == DST:
        sleep(2)

        # NOTE: since we don't launch any job selector workers in the background,
        # after triggering the creation of a backward job,
        # we expect the destination worker's job queue to have one job
        assert JobQueue.PENDING_JOBS.qsize() == 1

        backward_job = JobQueue.PENDING_JOBS.get()

        assert isinstance(backward_job, BackwardJob)


@pytest.mark.skip
def test_execute_a_backward_job_and_send_the_output():
    pass


def send_output_package_callback():
    pass
