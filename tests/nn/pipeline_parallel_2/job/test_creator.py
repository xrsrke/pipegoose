import pytest
import torch
from transformers import AutoModelForCausalLM

from pipegoose.nn.pipeline_parallel2._package import Package
from pipegoose.nn.pipeline_parallel2._job.creator import create_job
from pipegoose.nn.pipeline_parallel2._job.job_type import JobType
from pipegoose.nn.pipeline_parallel2._job.job import (
    ForwardJob,
    BackwardJob,
    JobStatus
)
from pipegoose.nn.pipeline_parallel2.queue import JobQueue
from pipegoose.nn.pipeline_parallel2._utils import sleep
from pipegoose.testing.utils import spawn, init_pipeline_context


@pytest.fixture(scope="session")
def module():
    MODEL_NAME = "bigscience/bloom-560m"
    return AutoModelForCausalLM.from_pretrained(MODEL_NAME)


def run_check_the_job_status_after_executing_a_job(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, package, module):
    pipeline_context = init_pipeline_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size,
        module, n_partitions=3, n_microbatches=5
    )
    job = create_job(package, pipeline_context)

    job.compute()

    assert job.status == JobStatus.EXECUTED


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
@pytest.mark.parametrize("package", ["forward_package", "backward_package"])
def test_the_job_status_after_executing_a_job(request, pipeline_parallel_size, package, module):
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
        module=module
    )


def run_execute_a_forward_job(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, package, module):
    pipeline_context = init_pipeline_context(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, module)
    forward_job = create_job(package, pipeline_context)

    ORIG_MICROBATCH_IDX = forward_job.input.metadata.microbatch_idx
    ORIG_PARTITION_IDX = forward_job.input.metadata.partition_idx
    # PARALLEL_MODE = ParallelMode.PIPELINE
    # SRC = parallel_context.get_global_rank()
    # LOCAL_RANK = parallel_context.get_local_rank(PARALLEL_MODE)
    # DST = parallel_context.get_next_local_rank(LOCAL_RANK, PARALLEL_MODE)

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
        assert getattr(output.metadata.training, key) == getattr(forward_job.input.metadata.training, key)

    # NOTE: we expect the metadata of the output package to
    # indicate which node executed it

    # TODO: update source rank and destination rank based on pipeline context
    # assert output.metadata.src == SRC
    # assert output.metadata.dst == DST


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
@pytest.mark.parametrize("package", ["forward_package"])
def test_execute_a_forward_job(request, pipeline_parallel_size, package, module):
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
        module=module
    )


def run_create_a_job_from_package(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, package, job_cls, module):
    pipeline_context = init_pipeline_context(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, module)

    job = create_job(package, pipeline_context)

    assert isinstance(job, job_cls)
    assert isinstance(job.key, str)
    assert callable(job.function) is True
    assert job.status == JobStatus.PENDING


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
@pytest.mark.parametrize("package, job_cls", [("forward_package", ForwardJob), ("backward_package", BackwardJob)])
def test_create_a_job_from_package(request, pipeline_parallel_size, package, job_cls, module):
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
        module=module
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
        forward_job = create_job(forward_package)

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
