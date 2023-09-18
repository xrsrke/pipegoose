from copy import deepcopy

import pytest
import torch
from torch import nn

from pipegoose.nn.pipeline_parallel2._job.backward import BackwardJob
from pipegoose.nn.pipeline_parallel2._job.creator import (
    create_job,
    schedule_backward_job,
)
from pipegoose.nn.pipeline_parallel2._job.forward import ForwardJob
from pipegoose.nn.pipeline_parallel2._job.job import JobStatus
from pipegoose.nn.pipeline_parallel2._job.job_type import JobType
from pipegoose.nn.pipeline_parallel2._package import Package
from pipegoose.nn.pipeline_parallel2._utils import sleep
from pipegoose.nn.pipeline_parallel2.pipeline_context import PipelineContext
from pipegoose.nn.pipeline_parallel2.queue import JobQueue
from pipegoose.nn.pipeline_parallel2.scheduler import SchedulerType, get_scheduler
from pipegoose.testing.utils import init_parallel_context, init_pipeline_context, spawn

# NOTE: use for creating a forward job
function = nn.Linear(2, 4)


@pytest.mark.parametrize("package", ["forward_package", "backward_package"])
def test_the_job_status_after_executing_a_job(request, package, pipeline_context):
    package = request.getfixturevalue(package)
    job = create_job(function, package, pipeline_context)

    job.compute()

    assert job.status == JobStatus.EXECUTED


@pytest.mark.parametrize("package", ["forward_package"])
def test_the_output_package_of_a_forward_job(request, package, pipeline_context):
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

    package = request.getfixturevalue(package)
    forward_job = create_job(function, package, pipeline_context)
    ORIG_MICROBATCH_IDX = forward_job.input.metadata.microbatch_idx
    ORIG_PARTITION_IDX = forward_job.input.metadata.partition_idx

    output = forward_job.compute()

    assert forward_job.output == output
    assert isinstance(output, Package)
    assert isinstance(output.data, torch.Tensor)
    assert output.metadata.job_type == JobType.FORWARD

    assert OUTPUT_DESTINATION[(ORIG_MICROBATCH_IDX, ORIG_PARTITION_IDX)] == (
        output.metadata.microbatch_idx,
        output.metadata.partition_idx,
    )
    for key in vars(output.metadata.training).keys():
        # TODO: add test automatically switch to create new package
        # for different mix precision training
        assert getattr(output.metadata.training, key) == getattr(forward_job.input.metadata.training, key)

    # NOTE: we expect the metadata of the output package to
    # indicate which node executed it
    # TODO: update source rank and destination rank based on pipeline context
    assert isinstance(output.metadata.src, int)
    assert isinstance(output.metadata.dst, int)


def run_forward_job_send_output_to_the_next_pipeline_stage(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, package
):
    pipeline_context = init_pipeline_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    forward_job = create_job(function, package, pipeline_context)

    forward_job.compute()

    if world_size > 1:
        from pipegoose.nn.pipeline_parallel2._comm import RECV_QUEUE

        sleep(5)
        assert RECV_QUEUE.qsize() == 1

        received_package = RECV_QUEUE.get()
        assert isinstance(received_package, Package)
        assert received_package.metadata.dst == rank


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2, 5])
@pytest.mark.parametrize("package", ["forward_package"])
def test_forward_job_send_output_to_the_next_pipeline_stage(request, pipeline_parallel_size, package):
    TENSOR_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    package = request.getfixturevalue(package)
    spawn(
        run_forward_job_send_output_to_the_next_pipeline_stage,
        world_size=pipeline_parallel_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=DATA_PARALLEL_SIZE,
        package=package,
    )


@pytest.mark.parametrize("package, job_cls", [("forward_package", ForwardJob), ("backward_package", BackwardJob)])
def test_create_a_job_from_package(request, package, job_cls, pipeline_context):
    package = request.getfixturevalue(package)
    job = create_job(function, package, pipeline_context)

    assert isinstance(job, job_cls)
    assert isinstance(job.key, str)
    assert callable(job.function) is True
    assert job.status == JobStatus.PENDING


def run_create_a_backward_job_if_a_tensor_do_backprop(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, forward_package
):
    SRC = forward_package.metadata.src
    N_PARTITIONS = 3
    N_MICROBATCHES = 5
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    scheduler = get_scheduler(SchedulerType.GPIPE)(N_MICROBATCHES, N_PARTITIONS)
    pipeline_context = PipelineContext(scheduler, parallel_context)
    rank = parallel_context.get_global_rank()

    # NOTE: both the forward job and backward job of the same package
    # execute on the same node
    if rank == SRC:
        ORIG_FORWARD_PACKAGE = deepcopy(forward_package)
        forward_package = schedule_backward_job(forward_package, pipeline_context)

        # NOTE: make sure we aren't change the package
        assert torch.equal(forward_package.data, ORIG_FORWARD_PACKAGE.data)
        assert forward_package.metadata == ORIG_FORWARD_PACKAGE.metadata

        data = forward_package.data
        data.sum().backward()

        sleep(2)

        # NOTE: since we don't launch any job selector workers in the background,
        # after triggering the creation of a backward job,
        # we expect the destination worker's job queue to have one job
        assert JobQueue.PENDING_JOBS.qsize() == 1

        backward_job = JobQueue.PENDING_JOBS.get()

        assert isinstance(backward_job, BackwardJob)


@pytest.mark.parametrize("pipeline_parallel_size", [2, 5])
def test_create_a_backward_job_if_a_tensor_do_backprop(forward_package, pipeline_parallel_size):
    TENSOR_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    spawn(
        run_create_a_backward_job_if_a_tensor_do_backprop,
        world_size=pipeline_parallel_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=DATA_PARALLEL_SIZE,
        forward_package=forward_package,
    )


@pytest.mark.skip
def test_execute_a_backward_job_and_send_the_output():
    pass


def test_execute_a_backward_job():
    pass
