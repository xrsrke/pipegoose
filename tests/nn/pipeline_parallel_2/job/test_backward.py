import time
from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
from torch import nn

from pipegoose.nn.pipeline_parallel2._comm import set_pipeline_context
from pipegoose.nn.pipeline_parallel2._job.backward import BackwardJob
from pipegoose.nn.pipeline_parallel2._job.creator import schedule_backward_job
from pipegoose.nn.pipeline_parallel2.pipeline_context import PipelineContext
from pipegoose.nn.pipeline_parallel2.queue import JobQueue, SavedActivation
from pipegoose.nn.pipeline_parallel2.scheduler import SchedulerType, get_scheduler
from pipegoose.testing.utils import init_parallel_context, spawn

# NOTE: use for creating a forward job
function = nn.Linear(2, 4)


@pytest.fixture
def forward_package_in_different_nodes(forward_package):
    return forward_package


@pytest.fixture
def forward_package_in_same_node(forward_package):
    forward_package.metadata.dst = forward_package.metadata.src
    return forward_package


def run_create_a_backward_job_if_a_tensor_do_backprop(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, forward_package
):
    SRC = forward_package.metadata.src
    DST = forward_package.metadata.dst
    N_PARTITIONS = 3
    N_MICROBATCHES = 5
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    scheduler = get_scheduler(SchedulerType.GPIPE)(N_MICROBATCHES, N_PARTITIONS)
    pipeline_context = PipelineContext(scheduler, parallel_context)
    set_pipeline_context(pipeline_context)
    rank = parallel_context.get_global_rank()

    dist.barrier()

    if rank == DST:
        # NOTE: we enqueue the backward job in the destination rank
        ORIG_FORWARD_PACKAGE = deepcopy(forward_package)
        forward_package = schedule_backward_job(forward_package, pipeline_context)

        # NOTE: make sure we aren't change the package
        assert torch.equal(forward_package.data, ORIG_FORWARD_PACKAGE.data)
        assert forward_package.metadata == ORIG_FORWARD_PACKAGE.metadata

        data = forward_package.data
        data.sum().backward()

        time.sleep(0.1)

        # NOTE: since we don't launch any job selector workers in the background,
        # after triggering the creation of a backward job,
        # we expect the destination worker's job queue to have one job
        assert JobQueue.PENDING_JOBS.qsize() == 1

        backward_job = JobQueue.PENDING_JOBS.get()
        assert isinstance(backward_job, BackwardJob)

    # NOTE: wait for the backward job to be created
    dist.barrier()
    time.sleep(0.1)

    if rank == SRC:
        assert JobQueue.PENDING_JOBS.qsize() == 0


@pytest.mark.parametrize("pipeline_parallel_size", [2, 5])
@pytest.mark.parametrize("package", ["forward_package_in_same_node", "forward_package_in_different_nodes"])
def test_create_a_backward_job_if_a_tensor_do_backprop_in_the_same_node(request, package, pipeline_parallel_size):
    TENSOR_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    forward_package = request.getfixturevalue(package)

    spawn(
        run_create_a_backward_job_if_a_tensor_do_backprop,
        world_size=pipeline_parallel_size,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=DATA_PARALLEL_SIZE,
        forward_package=forward_package,
    )


def test_execute_a_backward_job(backward_job):
    MICROBATCH_IDX = backward_job.input.metadata.microbatch_idx
    PARTITION_IDX = backward_job.input.metadata.partition_idx

    input = torch.randn_like(backward_job.input.data)
    linear = nn.Linear(input.shape[1], input.shape[0])
    output = linear(input).sum()
    # NOTE: set initial gradients for the backward job
    backward_job.input.data = torch.ones_like(output)

    output.backward(retain_graph=True)
    GROUND_W_GRAD = deepcopy(linear.weight.grad)
    GROUND_B_GRAD = deepcopy(linear.bias.grad)
    linear.weight.grad.zero_()
    linear.bias.grad.zero_()

    key = SavedActivation.get_key(MICROBATCH_IDX, PARTITION_IDX)
    SavedActivation.save_activations(key, output)

    _ = backward_job.compute()

    assert torch.equal(linear.weight.grad, GROUND_W_GRAD)
    assert torch.equal(linear.bias.grad, GROUND_B_GRAD)


@pytest.mark.skip
def test_execute_a_backward_job_and_send_the_output():
    pass
