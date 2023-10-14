# NOTE: Ideally, we should be able to test backward jobs without depending
# on running forward jobs. However, since backward jobs calculate
# the gradients with respect to the input of a forward job,
# to test backward jobs we need to run forward jobs first


import time
from copy import deepcopy

import pytest
import torch
from torch import nn

from pipegoose.nn.pipeline_parallel2._job.backward import BackwardJob
from pipegoose.nn.pipeline_parallel2._job.creator import schedule_backward_job
from pipegoose.nn.pipeline_parallel2._job.forward import (
    CreateForwardOutputPackageCallback,
    ForwardJob,
    SaveActivationIfTrainingCallback,
    SaveInputActivationsCallback,
)
from pipegoose.nn.pipeline_parallel2.queue import JobQueue
from pipegoose.testing.utils import init_pipeline_context, spawn


@pytest.fixture
def forward_job(forward_package, forward_function):
    """A forward job that set with callbacks that use in training. like save input activation and output activations for backward job"""
    callbacks = [SaveInputActivationsCallback, SaveActivationIfTrainingCallback]
    forward_job = ForwardJob(forward_function, forward_package, callbacks)
    return forward_job


@pytest.fixture
def forward_package_in_different_nodes(forward_package):
    return forward_package


@pytest.fixture
def forward_package_in_same_node(forward_package):
    forward_package.metadata.dst = forward_package.metadata.src
    return forward_package


def test_create_a_backward_job_if_a_tensor_do_backprop(forward_package, forward_function, parallel_context, pipeline_context):
    callbacks = [
        CreateForwardOutputPackageCallback(parallel_context, pipeline_context),
        SaveInputActivationsCallback,
        SaveActivationIfTrainingCallback,
    ]
    forward_job = ForwardJob(forward_function, forward_package, callbacks)

    # NOTE: we enqueue the backward job in the destination rank
    output = forward_job.compute()
    DATA = output.data.clone()
    METADATA = deepcopy(output.metadata)

    output = schedule_backward_job(output, pipeline_context)
    # NOTE: make sure we aren't change the package
    assert torch.equal(output.data, DATA)
    assert output.metadata == METADATA

    output.data.sum().backward()

    # NOTE: since we don't launch any job selector workers in the background,
    # after triggering the creation of a backward job,
    # we expect the destination worker's job queue to have one job
    time.sleep(0.1)
    assert JobQueue.PENDING_JOBS.qsize() == 1

    backward_job = JobQueue.PENDING_JOBS.get()
    assert isinstance(backward_job, BackwardJob)

    backward_job.compute()


# NOTE: scheduling backward job works, but this one didn't works
# no gradients in leaf tensors!
# @pytest.mark.skip(reason="TODO: fix this")
def test_execute_a_backward_job(forward_job, backward_package, pipeline_context):
    def function(*args, **kwargs):
        pass

    # output = forward_job.compute()
    # INITIAL_GRADS = torch.ones_like(output.data)
    # grad_package = Package(INITIAL_GRADS, forward_job.input.metadata)
    # grad_package.metadata.job_type = JobType.BACKWARD

    from pipegoose.nn.pipeline_parallel2._job.callback import Callback

    class SetupInputOutputActivations(Callback):
        def before_compute(self):
            from pipegoose.nn.pipeline_parallel2.queue import (
                save_input_activations,
                save_output_activations,
            )

            INPUT_SHAPE = (
                4,
                2,
            )
            LINEAR_SHAPE = (
                2,
                4,
            )
            input = torch.randn(*INPUT_SHAPE, requires_grad=True)
            linear = nn.Linear(*LINEAR_SHAPE)
            output = linear(input)

            MICROBATCH_IDX = self.job.input.metadata.microbatch_idx
            PARTITION_IDX = self.job.input.metadata.partition_idx
            save_input_activations(input, MICROBATCH_IDX, PARTITION_IDX)
            save_output_activations(output, MICROBATCH_IDX, PARTITION_IDX)

    callbacks = [SetupInputOutputActivations()]

    backward_job = BackwardJob(function, backward_package, callbacks)
    grads = backward_job.compute()

    assert isinstance(grads, torch.Tensor)


# def run_execute_a_backward_job_and_send_the_output(
#     rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, backward_package
# ):
#     def function():
#         pass

#     DST = 1
#     pipeline_context, parallel_context = init_pipeline_context(
#         rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
#     )
#     backward_job = BackwardJob(
#         function, backward_package, cbs=[CreateBackwardOutputPackageCallback()]
#     )
#     rank = parallel_context.get_global_rank()

#     dist.barrier()

#     if rank == DST:
#         backward_job.compute()

#         # assert torch.equal(leaf_tensor.grad, INPUT.grad)


# @pytest.mark.parametrize("pipeline_parallel_size", [2, 5])
# # @pytest.mark.parametrize("package", ["forward_package_in_same_node", "forward_package_in_different_nodes"])
# def test_execute_a_backward_job_and_send_the_output(backward_package, pipeline_parallel_size):
#     TENSOR_PARALLEL_SIZE = 1
#     DATA_PARALLEL_SIZE = 1

#     # forward_package = request.getfixturevalue(package)

#     spawn(
#         run_execute_a_backward_job_and_send_the_output,
#         world_size=pipeline_parallel_size,
#         tensor_parallel_size=TENSOR_PARALLEL_SIZE,
#         pipeline_parallel_size=pipeline_parallel_size,
#         data_parallel_size=DATA_PARALLEL_SIZE,
#         # forward_package=forward_package,
#         backward_package=backward_package,
#     )


def run_execute_a_backward_job_and_send_the_output(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, forward_package, forward_function
):
    pipeline_context, parallel_context = init_pipeline_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    callbacks = [
        CreateForwardOutputPackageCallback(parallel_context, pipeline_context),
        SaveInputActivationsCallback,
        SaveActivationIfTrainingCallback,
    ]
    forward_job = ForwardJob(forward_function, forward_package, callbacks)

    # NOTE: we enqueue the backward job in the destination rank
    forward_job.compute()
    output = schedule_backward_job(output, pipeline_context)
    output.sum().backward()
    backward_job = JobQueue.PENDING_JOBS.get()
    backward_job.compute()


def test_execute_a_backward_job_and_send_the_output(forward_package, forward_function):
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 2
    DATA_PARALLEL_SIZE = 1

    WORLD_SIZE = TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE * DATA_PARALLEL_SIZE

    spawn(
        run_execute_a_backward_job_and_send_the_output,
        world_size=WORLD_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        forward_package=forward_package,
        forward_function=forward_function,
    )
