import pytest
import torch
from torch import nn

from pipegoose.nn.pipeline_parallel2._job.creator import create_job
from pipegoose.nn.pipeline_parallel2._job.job_type import JobType
from pipegoose.nn.pipeline_parallel2._package import Package
from pipegoose.nn.pipeline_parallel2._utils import sleep
from pipegoose.testing.utils import init_pipeline_context, spawn

# NOTE: use for creating a forward job
function = nn.Linear(2, 4)


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


def test_forward_job_save_activations_for_backward_pass(forward_package, pipeline_context):
    forward_job = create_job(function, forward_package, pipeline_context)

    output = forward_job.compute()

    from pipegoose.nn.pipeline_parallel2._job.forward import get_activation_name
    from pipegoose.nn.pipeline_parallel2.queue import get_saved_activations

    name = get_activation_name(forward_package.metadata.microbatch_idx, forward_package.metadata.partition_idx)

    saved_activations = get_saved_activations(name)

    assert isinstance(saved_activations, torch.Tensor)
    assert torch.equal(saved_activations, output.data)


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
