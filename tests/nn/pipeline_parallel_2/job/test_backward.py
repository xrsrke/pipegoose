import time
from copy import deepcopy

import pytest
import torch
from torch import nn

from pipegoose.nn.pipeline_parallel2._job.backward import (
    BackwardJob,
    CreateBackwardOutputPackageCallback,
)
from pipegoose.nn.pipeline_parallel2._job.callback import Callback
from pipegoose.nn.pipeline_parallel2._job.creator import schedule_backward_job
from pipegoose.nn.pipeline_parallel2._job.forward import (
    CreateForwardOutputPackageCallback,
    ForwardJob,
    SaveActivationIfTrainingCallback,
    SaveInputActivationsCallback,
)
from pipegoose.nn.pipeline_parallel2._job.job_type import JobType
from pipegoose.nn.pipeline_parallel2._package import Package
from pipegoose.nn.pipeline_parallel2.queue import JobQueue
from pipegoose.testing.utils import init_pipeline_context, spawn


class SetupInputOutputActivations(Callback):
    """Manually setup the input and output activations for backward job.

    NOTE: this typically done by the forward job, but here we want to test
    the backward job in isolation so we manually setup the input
    and output activations.
    """

    def __init__(self, input, output):
        self.input = input
        self.output = output

    def before_compute(self):
        from pipegoose.nn.pipeline_parallel2.queue import (
            save_input_activations,
            save_output_activations,
        )

        MICROBATCH_IDX = self.job.input.metadata.microbatch_idx
        PARTITION_IDX = self.job.input.metadata.partition_idx
        save_input_activations(self.input, MICROBATCH_IDX, PARTITION_IDX)
        save_output_activations(self.output, MICROBATCH_IDX, PARTITION_IDX)


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


def test_the_gradient_output_of_a_backward_job(backward_package):
    def function(*args, **kwargs):
        pass

    HIDDEN_SIZE = 2
    INPUT_SHAPE = (
        backward_package.data.shape[0],
        HIDDEN_SIZE,
    )
    LINEAR_SHAPE = (
        HIDDEN_SIZE,
        backward_package.data.shape[-1],
    )

    input = torch.randn(*INPUT_SHAPE, requires_grad=True)
    linear = nn.Linear(*LINEAR_SHAPE)
    output = linear(input)
    ORIG_INPUT = deepcopy(input)
    ORIG_LINEAR = deepcopy(linear)

    REF_OUTPUT = ORIG_LINEAR(ORIG_INPUT)
    INITIAL_GRADS = torch.ones_like(REF_OUTPUT)
    torch.autograd.backward(REF_OUTPUT, INITIAL_GRADS)

    callbacks = [SetupInputOutputActivations(input, output)]

    backward_job = BackwardJob(function, backward_package, callbacks)
    grads = backward_job.compute()

    assert isinstance(grads, torch.Tensor)
    assert torch.allclose(grads, ORIG_INPUT.grad)


def run_check_the_destination_of_output_package_from_a_backward_job(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, backward_package
):
    def function(*args, **kwargs):
        pass

    HIDDEN_SIZE = 2
    INPUT_SHAPE = (
        backward_package.data.shape[0],
        HIDDEN_SIZE,
    )
    LINEAR_SHAPE = (
        HIDDEN_SIZE,
        backward_package.data.shape[-1],
    )

    pipeline_context, parallel_context = init_pipeline_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    input = torch.randn(*INPUT_SHAPE, requires_grad=True)
    linear = nn.Linear(*LINEAR_SHAPE)
    output = linear(input)

    callbacks = [
        SetupInputOutputActivations(input, output),
        CreateBackwardOutputPackageCallback(parallel_context, pipeline_context),
    ]

    backward_job = BackwardJob(function, backward_package, callbacks)
    output = backward_job.compute()

    assert backward_job.output == output
    assert isinstance(output, Package)
    assert isinstance(output.data, torch.Tensor)
    assert output.metadata.job_type == JobType.BACKWARD

    for key in vars(output.metadata.training).keys():
        assert getattr(output.metadata.training, key) == getattr(backward_job.input.metadata.training, key)


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2, 4])
def test_the_destination_of_output_package_from_a_backward_job(pipeline_parallel_size, backward_package):
    TENSOR_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    WORLD_SIZE = TENSOR_PARALLEL_SIZE * pipeline_parallel_size * DATA_PARALLEL_SIZE

    spawn(
        run_check_the_destination_of_output_package_from_a_backward_job,
        world_size=WORLD_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=DATA_PARALLEL_SIZE,
        backward_package=backward_package,
    )
